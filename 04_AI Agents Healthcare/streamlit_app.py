import os
import subprocess
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from graph.graph_agentPatient import app

def main():
    st.title("Medical Query Assistants")

    # Doctor Agent Section
    st.header("Doctor's Query Assistant")
    doctor_question = st.text_input(
        "Enter your question about the medical query:",
        "Based on the Diabetes of John Doe, recommend me the treatment plan for him?"
    )
    doctor_loop = st.text_input(
        "Want to learn more about the symptoms and its cure? (yes/no):",
        "no"
    )
    doctor_medical = ""
    if doctor_loop.lower() == "yes":
        doctor_medical = st.text_input("Enter the medical term:", "diabetes")

    if st.button("Get Doctor's Response"):
        os.environ["QUESTION_OVERRIDE"] = doctor_question
        os.environ["HUMAN_IN_LOOP_OVERRIDE"] = doctor_loop
        os.environ["MEDICAL_NAME_OVERRIDE"] = doctor_medical

        cmd = ["python", "main.py"]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            st.error(f"Error: {stderr}")
        else:
            st.text_area("Doctor's Output", value=stdout, height=200)

    # Patient Agent Section
    st.header("Patient's Query Assistant")
    dietary_preference = st.radio(
        "Select your dietary preference:",
        ("Vegetarian", "Non-Vegetarian")
    )
    patient_question = st.text_input("Enter your question:")

    if st.button("Get Patient's Recommendation"):
        if patient_question:
            # Append dietary preference to the user's question
            if dietary_preference == "Vegetarian":
                full_question = f"I am vegetarian, {patient_question}"
            else:
                full_question = f"I am non-vegetarian, {patient_question}"

            # Invoke the ML model with the full question
            result = app.invoke(input={"question": full_question})

            # Display the result
            st.write("Recommendation:")
            st.write(result['generation'])
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()