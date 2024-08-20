<!-- ABOUT THE PROJECT -->
## Feature Description
<p>This feature vets through the list of common mistakes provided and generates corresponding messages to describe the difference between the responses and the answers.</p>
It is driven mainly by AST pattern matching of the Sympy-parsed representations of the responses and answers.
The function and its dependencies are hosted as a docker image on ECR and is deployed on AWS lambda. 

## Usage
The input to the function is in this JSON format:
```
{
  "commonMistakes": [
    {
      "submission": "4",
      "latexSubmission": "5",
      "simplifiedSubmission": "5",
      "frequency": 6,
      "feedback": "Incorrect",
      "mark": 0,
      "color": "",
      "responseAreaId": "655f95f1-c606-41a0-8905-a8baaa4a8a9f",
      "universalResponseAreaId": "da886822-618a-4c32-a332-18b0783e33dc",
      "params": {
        "strict_syntax": false,
        "symbols": {}
      },
      "answer": "-4"
    },
    {
      "submission": "3",
      "latexSubmission": "3",
      "simplifiedSubmission": "3",
      "frequency": 5,
      "feedback": "Incorrect",
      "mark": 0,
      "color": "",
      "responseAreaId": "655f95f1-c606-41a0-8905-a8baaa4a8a9f",
      "universalResponseAreaId": "da886822-618a-4c32-a332-18b0783e33dc",
      "params": {
        "strict_syntax": false,
        "symbols": {}
      },
      "answer": "4"
    }]
}
```

The function returns data in the following format:
```
{
  "statusCode": 200,
  "body": [
    {
      "submission": "4",
      "latexSubmission": "5",
      "simplifiedSubmission": "5",
      "frequency": 6,
      "feedback": "Incorrect",
      "mark": 0,
      "color": "",
      "responseAreaId": "655f95f1-c606-41a0-8905-a8baaa4a8a9f",
      "universalResponseAreaId": "da886822-618a-4c32-a332-18b0783e33dc",
      "params": {
        "strict_syntax": false,
        "symbols": {
          "lamda": {
            "latex": "\\lambda",
            "aliases": ["lambda", "lambda"]
          }
        }
      },
      "answer": "-4",
      "recommendedFeedback": "(1) The student's response differs by the term -."
    },
    {
      "submission": "3",
      "latexSubmission": "3",
      "simplifiedSubmission": "3",
      "frequency": 5,
      "feedback": "Incorrect",
      "mark": 0,
      "color": "",
      "responseAreaId": "655f95f1-c606-41a0-8905-a8baaa4a8a9f",
      "universalResponseAreaId": "da886822-618a-4c32-a332-18b0783e33dc",
      "params": {
        "strict_syntax": false,
        "symbols": {
          "lamda": {
            "latex": "\\lambda",
            "aliases": ["lambda", "lambda"]
          }
        }
      },
      "answer": "4"
    }]
}
```


### Dependencies

<ul>
  <li>sympy==1.12.1</li>
  <li>ppython-Levenshtein==0.23.0</li>
  <li>numpy==1.26.1</li>
  <li>zss==1.2.0</li>
</ul>

Additional dependencies included in the docker file includes Karl's evaluationFunction and also a modified implementation of the compare.py from the zss library.







