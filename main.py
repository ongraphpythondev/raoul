from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
load_dotenv()
openai_api=st.secrets["OPEN_AI_API"]
# openai_api=os.getenv("Open_ai_api")


st.title("Issue Classifier")
text=st.text_area("Text",height=300)
button=st.button("submit",use_container_width=True)

if button:
# Prompt Template
    template='''{email}
        
        Your are an email classifier with experience of 20 years.In the above email find the catagory categorizes the customer query, based on its content and keywords.

        the output will be like below example

        Example Category:["billing","technical issue","product inquiry","callback",","customer dissatisfaction","gas meter","personal account information","Technical issue internet","contract change internet"]
        if it contain more then one category return all categories
        you can also give your own category.
        
        
        output does not contain Answer
        '''
    print(openai_api)

    categorization_template=PromptTemplate(
        input_variables=["email"],
        template=template,
        validate_template=False
    )
    
    llm_model=OpenAI(openai_api_key=openai_api,
                     temperature=0)
    category_chain=LLMChain(llm=llm_model,prompt=categorization_template,verbose=True)
    
    
    output=category_chain.run(text)
    
    print("output",output)
    st.write(output)