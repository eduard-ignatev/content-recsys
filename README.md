# content-recsys
This was the final project assignment in [StartML](https://karpov.courses/ml-start) specialization from [Karpov.Courses](https://karpov.courses/). The repository serves to demonstrate my skills in Machine Learning system development.

## Assignment
The task was to build a social network posts recommender system. Raw data included users, posts and feed data from PostgreSQL.

The project consisted of three parts:
1. Develop the RecSys service using classical Machine Learning
2. Improve the RecSys service using Deep Learning techniques
3. Run an A/B testing experiment with two versions of the service

#### API specification
**Endpoint** - GET /post/recommendations/
| Parameter | Overview |
| --------- | -------- |
| id        | User ID that request recommendations |
| time      | Datetime type object |
| limit     | Qty of posts for recommendation |

## Implementation
#### Content RecSys Baseline:
[`content-recsys-notebook.ipynb`](Content%20RecSys%20Baseline/content-recsys-notebook.ipynb) - EDA, feature engineering, model training

[`content-recsys.py`](Content%20RecSys%20Baseline/content-recsys.py) - service code

#### Content RecSys Deep Learning:
[`content-recsys-dl-notebook.ipynb`](Content%20RecSys%20Deep%20Learning/content-recsys-dl-notebook.ipynb) - feature engineering, model training

[`content-recsys-dl.py`](Content%20RecSys%20Deep%20Learning/content-recsys-dl.py) - service code

#### Content RecSys AB Testing:
[`content-recsys-ab-testing.py`](Content%20RecSys%20AB%20Testing/content-recsys-ab-testing.py) - service code

[`content-recsys-ab-testing-notebook.ipynb`](Content%20RecSys%20AB%20Testing/content-recsys-ab-testing-notebook.ipynb) - stat tests, analysis

## Demonstration
Using Uvicorn to start the server process:
![Server startup](https://user-images.githubusercontent.com/97912967/194586881-e7c4170a-bd62-4769-bf83-5dfca8858717.png)

Checking GET request with Postman:
![GET request](https://user-images.githubusercontent.com/97912967/194586922-7e57e76a-ebc0-4300-b147-dd20cf77779c.png)

Service provides corresponsing errors if user ID doesn't exist or in case of a type error:
![Responses](https://user-images.githubusercontent.com/97912967/194586958-756cdb3e-f533-4718-b8e5-90d14dbc2fee.png)
