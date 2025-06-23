# Tesla HR FAQs Chatbot

A Streamlit application that helps employees understand Tesla's policies and benefits through an AI-powered chatbot interface.

## Features

- AI-powered responses to HR policy questions
- Tesla-branded UI with external CSS styling
- Chat history tracking
- Source citation for answers
- Example questions for easy starting points

## Deployment Instructions

### Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables in `.env` file:
   ```
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_API_VERSION=your_api_version
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment
   AZURE_OPENAI_CHAT_DEPLOYMENT=your_chat_deployment
   ```

3. Run the application:
   ```
   streamlit run main.py
   ```

### Streamlit Cloud Deployment

1. Push your code to a GitHub repository
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your GitHub repository
4. Set the main file path to `main.py`
5. Add your environment variables in the Streamlit Cloud secrets management
6. Deploy the application

## Environment Variables

The following environment variables need to be set in Streamlit Cloud:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`

## Data Storage

The application can use either local storage or Airtable for chat history. Configure Airtable credentials in the `.env` file if needed.

## Project Structure

- `main.py`: Main application file
- `ingest.py`: Script to ingest and embed documents
- `setup_airtable.py`: Script to set up Airtable integration
- `airtable_integration.py`: Functions to interact with Airtable
- `airtable_client.py`: Low-level Airtable API client
- `static/styles.css`: Tesla-branded UI styling
- `.streamlit/config.toml`: Streamlit configuration
- `requirements.txt`: Python dependencies
