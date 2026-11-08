"""Reproducible baseline training with disjoint evaluation and no code execution."""
import ast
import hashlib
import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from plagiarism_detector.ml import ROOT, lexical, pair_features
from plagiarism_detector.core import FINGERPRINT_POLICY

def metrics(y,p):
    pred=[int(v>=.5) for v in p]
    pr,re,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    accepted=[i for i,v in enumerate(p) if v<=.3 or v>=.7]
    return dict(precision=float(pr),recall=float(re),f1=float(f1),roc_auc=float(roc_auc_score(y,p)),false_positive_rate=float(fp/max(1,fp+tn)),confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]],decision_threshold=.5,verdict_thresholds=[.3,.7],verdict_coverage=len(accepted)/len(y),decisive_accuracy=sum(pred[i]==y[i] for i in accepted)/max(1,len(accepted)))

def rows(split,seen):
    out=[]
    for path in sorted((ROOT/'data').glob(split+'_*.json')):
        for entry in json.loads(path.read_text())['rows']:
            if entry.get('truncated_cells'): continue
            row=entry['row']; code=row['code']
            if len(code)>18000 or len(code)<100: continue
            try: text=lexical(code)
            except (SyntaxError,ValueError,RecursionError): continue
            # AST-normalised structural dedup also blocks renamed clones across splits.
            from plagiarism_detector.core import ast_stream
            key=hashlib.sha256(' '.join(ast_stream(code)).encode()).hexdigest()
            if key in seen: continue
            seen.add(key)
            out.append(dict(code=code,text=text,label=int(row['source']!='Human'),source=row['source'],key=key))
    return out

class Rename(ast.NodeTransformer):
    def visit_Name(self,node):
        if not node.id.startswith('__'):
            node.id='v_'+node.id
        return node
    def visit_arg(self,node):
        node.arg='v_'+node.arg
        return node

def make_pairs(samples):
    # Related examples are controlled synthetic transformations, not real plagiarism labels.
    codes=[r['code'] for r in samples if r['label']==0 and len(r['code'])<=6000][:120]
    x=[]; y=[]
    rng=random.Random(42)
    for i,code in enumerate(codes):
        renamed=ast.unparse(Rename().visit(ast.parse(code)))
        for variant in ('# formatting variant\n'+ast.unparse(ast.parse(code)),renamed):
            x.append(pair_features(code,variant)); y.append(1)
        others=[j for j in range(len(codes)) if j!=i]
        for j in rng.sample(others,min(2,len(others))):
            x.append(pair_features(code,codes[j])); y.append(0)
    return x,y,len(codes)

def main():
    seen=set(); train=rows('train',seen); test=rows('test',seen)
    if any(sum(r['label']==label for r in data)<20 for data in (train,test) for label in (0,1)):
        raise ValueError('Need at least 20 valid distinct examples per class in each split')
    model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),max_features=20000,sublinear_tf=True,token_pattern=r'\S+'),LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42))
    model.fit([r['text'] for r in train],[r['label'] for r in train])
    ai=metrics([r['label'] for r in test],model.predict_proba([r['text'] for r in test])[:,1])
    (ROOT/'models').mkdir(exist_ok=True)
    with (ROOT/'models/authorship.pkl').open('wb') as f: pickle.dump(model,f)
    print('Authorship complete:',ai,flush=True)
    x,y,n=make_pairs(train); xt,yt,nt=make_pairs(test)
    copy=RandomForestClassifier(n_estimators=150,max_depth=6,min_samples_leaf=3,class_weight='balanced',random_state=42)
    copy.fit(x,y)
    copy.fingerprint_policy_ = FINGERPRINT_POLICY
    cm=metrics(yt,copy.predict_proba(xt)[:,1])
    with (ROOT/'models/copy.pkl').open('wb') as f: pickle.dump(copy,f)
    report=dict(fingerprint_policy=FINGERPRINT_POLICY,seed=42,sklearn_version=sklearn.__version__,authorship=dict(train=len(train),test=len(test),train_sources=dict(Counter(r['source'] for r in train)),test_sources=dict(Counter(r['source'] for r in test)),metrics=ai),copy=dict(train_bases=n,test_bases=nt,train_pairs=len(y),test_pairs=len(yt),metrics=cm,feature_names=['token_jaccard','ast_jaccard','token_length_ratio','ast_histogram_overlap','token_sequence_ratio']),limitations=['Small convenience subset, not representative of classroom submissions.','Authorship scores are uncalibrated model outputs, not probabilities of misconduct.','Official train/test split plus structural dedup; repository/problem identities unavailable, so related-task leakage remains possible.','No held-out generator evaluation; do not generalize to unseen models.','Copy positives are synthetic formatting and renaming transformations; random negatives may contain unknown related code. No validated same-assignment hard-negative benchmark.','Automatic verdicts include an uncertainty band; neither model proves copying or authorship.'])
    (ROOT/'models/report.json').write_text(json.dumps(report,indent=2))
    (ROOT/'models/model-card.md').write_text('# Experimental classifiers\n\nTraining source: https://huggingface.co/datasets/HanxiGuo/CodeMirage (CC-BY-NC-ND-4.0). Noncommercial research/teaching baseline; check dataset and underlying-code terms before redistribution.\n\nAuthorship: lexical TF-IDF + logistic regression. Copying: five pairwise features + Random Forest. Official test rows are held out; structural duplicates are removed before training. See report.json for actual counts and metrics.\n\n'+ '\n'.join('- '+s for s in report['limitations'])+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
