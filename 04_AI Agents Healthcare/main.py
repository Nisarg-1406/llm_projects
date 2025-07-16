import os
from dotenv import load_dotenv
load_dotenv()

from graph.graph_agentDoctor import app

# if __name__ == "__main__":
#     result = app.invoke(input={"question": "Based on the Diabetes of John Doe, recommend me the treatment plan for him?"})
#     # result = app.invoke(input={"question": "Based on Myopia of John Doe, recommend me the treatment plan for him?"})
#     # result = app.invoke(input={"question": "John Doe is facing with High Temperature, recommend me the treatment plan for him?"})
#     # print("RESULT: ", result)
#     print(result['generation'])

question_text = os.environ.get("QUESTION_OVERRIDE") or "Based on the Diabetes of John Doe..."
try:
    result = app.invoke(input={"question": question_text})
    print(result['generation'])
except Exception as e:
    print("Please Try Again!!")

# from graph.graph_agentPatient import app

# if __name__ == "__main__":
#     result = app.invoke(input={"question": "Raj is vegitarian and has High Cholesterol, recommend him the nutrition plan?"})
#     # result = app.invoke(input={"question": "What all food items can be made of Rajma?"})
#     # result = app.invoke(input={"question": "John Doe is facing with High Temperature, recommend me the treatment plan for him?"})
#     print(result['generation'])
