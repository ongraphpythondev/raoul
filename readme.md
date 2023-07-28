
## Create python environment

`python -m venv venv`


## Activate environment 

### Linux
`source venv\bin\activate`
### windows
`venv\Script\activate`

## Add Your OpenAI Api key in .env.example file and rename it .env
`Open_ai_api= "YOUR API KEY"`

## Run the Steamlit app
`streamlit run app.py`

### if you got error
`streamlit run app.py --server.fileWatcherType none`