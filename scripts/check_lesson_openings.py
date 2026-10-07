"""Guard the learner's preference without deleting model retrieval or exercises."""
import unittest
from lesson_openings import strip_html,strip_markdown
class OpeningPolicy(unittest.TestCase):
 def test_preserve_objective_and_worked_example(self):
  source='<h2>Start cold · recall</h2><div id="warmup"></div><p>Close your notes. What is a node?</p><p>Your tangible win: trace a graph.</p><h2>Worked example</h2><pre>x = 2</pre>'
  result=strip_html(source)
  self.assertNotIn('warmup',result);self.assertNotIn('What is a node?',result)
  self.assertIn('Your tangible win',result);self.assertIn('<pre>x = 2</pre>',result)
 def test_keep_model_retrieval(self):
  source='<h2>A retrieval baseline sets a stronger flat-table bar</h2><p>Retrieve neighbors from the support set.</p><h2>Exercises</h2><p>Predict the result before running this code.</p>'
  self.assertEqual(strip_html(source),source)
 def test_keep_attached_prerequisites(self):
  source='<p><strong>Recall before reading:</strong> What is a fold? <strong>Prerequisites:</strong> grouped validation.</p>'
  result=strip_html(source)
  self.assertNotIn('What is a fold',result);self.assertIn('<strong>Prerequisites:</strong> grouped validation.',result)
 def test_markdown_code_is_untouched(self):
  source='## Retrieve before reading\n\nWithout notes: what is X?\n\n[[WARMUP]]\n\n## Exercise\n\n```python\nx = 1\n```\n'
  result=strip_markdown(source)
  self.assertNotIn('what is X?',result);self.assertIn('```python\nx = 1\n```',result)
  self.assertEqual(strip_markdown(result),result)
 def test_static_review_section(self):
  source='<section id="retrieval"><h2>Retrieve before reading</h2><p>Why?</p></section><section id="model"><p>Model content</p></section>'
  self.assertEqual(strip_html(source),'<section id="model"><p>Model content</p></section>')
if __name__=='__main__':unittest.main()
