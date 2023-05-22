from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
load_dotenv()
# openai_api=st.secrets["openai_api_key"]
openai_api=os.getenv("openai_api_key")


st.title("Issue Classifier")
text=st.text_area("Text",height=300)
button=st.button("submit",use_container_width=True)

if button:
# Prompt Template
    template='''{email}
        
        Your are an email classifier with experience of 20 years.In the above email find the catagory categorizes the customer query, based on its content and keywords.

        First Level Category: [Account Management]
        Second Level Categories: [Account Creation,Account Deletion,Password Reset,Update Contact Information]

        First Level Category: [Billing & Payments]
        Second Level Categories: [Invoice Discrepancy,Late Payment,Payment Failure,Installments Inquiry,Unrecognized Charge,Refund Request]	

        First Level Category: [Product & Service Issues]
        Second Level Categories: [Internet Speed,Gas Meter Reading,Electricity Outage,Water Leakage]

        First Level Category: [Account Upgrades]
        Second Level Categories: [Service Upgrade Inquiry,Electricity Plan Change Inquiry]

        First Level Category: [Tariffs & Pricing]
        Second Level Categories: [Tariff Inquiry,Price Break Query]

        First Level Category: [Callback Request]
        Second Level Categories: [--]

        First Level Category: [Customer Dissatisfaction]
        Second Level Categories: [Response Time,Service Quality,Billing Issues,Account Management Issues]

        the output will be like below:
        Output: write the name First Level Category
                write the name Second Level Category

        Example Outputs 1: Account Management
                           Account Creation 
        

        Example Outputs 1: Billing & Payments
                           Payment Failure,Refund Request
                            

        Example Outputs 1: Account Management,Callback Request,Customer Dissatisfaction
                            Account Deletion,Response Time,Service Quality
                           
                                                      
        Note: if it contain more then one category return all categories
        
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




