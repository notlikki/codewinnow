import http.client
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode
from plagiarism_detector.web import DetectorHandler
from plagiarism_detector.ml import ROOT, detect

class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),DetectorHandler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def post(self,fields):
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=30)
        connection.request('POST','/',urlencode(fields),{'Content-Type':'application/x-www-form-urlencoded'})
        response=connection.getresponse(); status=response.status; body=response.read().decode();connection.close()
        return status,body
    def test_incomplete_python_returns_error(self):
        status,body=self.post({'left':'x = (','right':'pass'})
        self.assertEqual(status,400)
        self.assertIn("class='error'",body)
    @unittest.skipUnless((ROOT/'models/copy.pkl').exists(),'Train models first')
    def test_trained_models_in_web(self):
        code=(ROOT/'examples/ml_demo.py').read_text()
        status,body=self.post({'left':code,'right':code})
        self.assertEqual(status,200)
        self.assertIn('SAME CODE',body)
        self.assertIn('Token similarity',body)
        self.assertIn('AST similarity',body)
    @unittest.skipUnless((ROOT/'models/authorship.pkl').exists(),'Train models first')
    def test_authorship_without_reference(self):
        code=(ROOT/'examples/ml_demo.py').read_text()
        result=detect(code)
        self.assertIn('authorship_a',result)
        self.assertNotIn('copy',result)
        self.assertTrue(0<=result['authorship_a']['ai_score']<=1)
