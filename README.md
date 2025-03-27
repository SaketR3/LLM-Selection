Request:
GET https://llm-selection.onrender.com/select-llms?question=superbowlwinner

Returns: 
{
  "llms": [
    "Gemini",
    "4o",
    "R1"
  ],
  "topic": "Sports"
}


Request:
GET https://llm-selection.onrender.com/update-system?llm=4o&topic=Sports

Returns: 
{
  "topic": "Sports"
}
