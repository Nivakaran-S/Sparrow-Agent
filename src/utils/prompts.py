clarification_with_user_instructions = """
These are the messages that have been exchanged so far regarding the user's parcel request or tracking inquiry:
<Messages>
{messages}
</Messages>

Today's date is {date}.

Assess whether you need to ask a clarifying question, or if the user has already provided enough information for you to proceed with their parcel request or tracking task.
IMPORTANT: If you can see in the messages history that you have already asked a clarifying question, you almost always do not need to ask another one. Only ask another question if ABSOLUTELY NECESSARY.

If there are abbreviations, shipment codes, or terms related to parcels or logistics that are unclear, ask the user to clarify.
If you need to ask a question, follow these guidelines:
- Be concise while gathering all necessary information to process the parcel request.
- Make sure to collect all information needed to track, consolidate, or manage the shipment in a clear and structured manner.
- Use bullet points or numbered lists if appropriate for clarity. Ensure this uses markdown formatting and will render correctly if passed to a markdown renderer.
- Do not ask for unnecessary information or information the user has already provided.

Respond in valid JSON format with these exact keys:
"need_clarification": boolean,
"question": "<question to ask the user to clarify their parcel request>",
"verification": "<verification message that we will start processing the parcel request>"

If you need to ask a clarifying question, return:
"need_clarification": true,
"question": "<your clarifying question>",
"verification": ""

If you do not need to ask a clarifying question, return:
"need_clarification": false,
"question": "",
"verification": "<acknowledgement message that you will now start processing or tracking the parcel>"

For the verification message when no clarification is needed:
- Acknowledge that you have sufficient information to proceed
- Briefly summarize the key aspects of the parcel request (e.g., shipment details, tracking numbers, consolidation instructions)
- Confirm that you will now begin processing or tracking the shipment
- Keep the message concise and professional
"""

transform_messages_into_customer_query_brief_prompt = """
You will be given a set of messages that have been exchanged so far between yourself and the user.  
Your job is to translate these messages into a detailed and actionable parcel request brief that will be used to guide parcel processing, tracking, or consolidation.

The messages that have been exchanged so far between yourself and the user are:
<Messages>
{messages}
</Messages>

Today's date is {date}.

You will return a single, clear, and actionable brief that summarizes the user's parcel request.

Guidelines:
1. Maximize Specificity and Detail
- Include all known shipment details provided by the user (e.g., parcel dimensions, weight, destination, delivery preferences, tracking numbers).
- Explicitly list any key attributes or instructions mentioned by the user for consolidation, delivery timing, or handling.

2. Handle Unstated Dimensions Carefully
- If certain aspects are not specified but are critical for processing (e.g., preferred courier, insurance, or packaging requirements), note them as open considerations rather than making assumptions.
- Example: Instead of assuming “fast delivery,” say “consider delivery options unless the user specifies a preference.”

3. Avoid Unwarranted Assumptions
- Never invent shipment details, preferences, or constraints that the user hasn’t stated.
- If certain details are missing (e.g., shipment value, fragile contents), explicitly note this lack of specification.

4. Distinguish Between Required Actions and Optional Preferences
- Required actions: Steps or details necessary for successful processing or tracking of the shipment.
- User preferences: Specific instructions provided by the user (must only include what the user stated).
- Example: “Process and track parcels to the provided address, prioritizing delivery timing as specified by the user.”

5. Use the First Person
- Phrase the brief from the perspective of the user.

6. Sources / References
- If tracking numbers or courier platforms are mentioned, reference the official tracking pages or courier portals.
- If specific shipment services or rules apply, prioritize official service guidelines over general logistics advice.
"""

