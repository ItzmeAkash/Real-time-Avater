#!/usr/bin/env python3
"""
Script to evaluate a conversation transcript and display results in terminal.
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("evaluator")

# Import the evaluation function from main.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import evaluate_conversation


def parse_transcript(transcript_text: str) -> List[Dict[str, str]]:
    """
    Parse a conversation transcript text into the format expected by evaluate_conversation.

    Expected format:
    Customer: message
    Salesperson: message
    ...
    """
    transcript = []
    lines = transcript_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if line starts with "Customer:" or "Salesperson:"
        if line.startswith("Customer:"):
            message = line.replace("Customer:", "").strip()
            transcript.append({
                "role": "Customer",
                "message": message,
                "timestamp": datetime.now().isoformat(),
            })
        elif line.startswith("Salesperson:"):
            message = line.replace("Salesperson:", "").strip()
            transcript.append({
                "role": "Salesperson",
                "message": message,
                "timestamp": datetime.now().isoformat(),
            })

    return transcript


async def main():
    """Main function to run the evaluation"""

    # The conversation transcript provided by the user
    conversation_transcript_text = """Customer: Look, I've already wasted 10 days with another bank. They promised quick approval and did nothing. Don't tell me you will also take forever.
Salesperson: Sorry to hear that .. let me see how best I can help you
Customer: See… this is exactly what everyone says in the beginning. "Let me see, let me check, give me some time." I'm honestly fed up now. I need this loan fast. The bike offer ends in three days. How do I know you won't also delay like the other bank?
Salesperson: Give me a min to check the system and see what is the real issue .. may i put you on hold for a min
Customer: Hold? Again? See, this is what frustrates me. Every bank keeps putting me on hold and nothing actually moves. Alright… fine. One minute. But please don't come back with the same "sir your case is under process" line.
Salesperson: I checked the system, there is a mismatch in your name to complete the KYC. The name provided in Aadhaar and your salary slip does not match and the system does allow to proceed further on it
Customer: What? Seriously? After all this running around, now you're telling me even my name is a problem? Rajesh Sharma is Rajesh Sharma only. How can this small thing stop the whole loan? So what does this actually mean for me now? Are you saying this will delay everything again?
Salesperson: Sir the Salary slip the name is Rajesh Sarma - H is missing. Can you reshare the salary slip
Customer: Arre… yaar. This is exactly what I was afraid of. Such a small spelling mistake and everything gets stuck. But fine…tell me clearly—if I send you the corrected salary slip now, will this move fast? Or is there some other hidden issue waiting to pop up?
Salesperson: Yes, once i get the salary slip, I will personally ensure within 24 hours your loan will be sanctioned and you can take the most valued bike .. we are here to help you get your dreams fulfilled sir
Customer: See… this is where I get nervous. Everyone promises "24 hours" and "personally ensure," but when the time comes, nobody picks the call. I'm asking you straight — can you commit that there won't be any more surprises after this salary slip?
Salesperson: No sir, I checked - there is only one red flag in your KYC. Once KYC is approved, I will ensure all the steps are completed in 4 hours and you will get notifications from the system
Customer: Hmm… okay. At least now you're sounding a bit more concrete. If you're saying this is the ONLY issue, fine… I'll resend the salary slip. But please understand — I'm depending on you.
Salesperson: Dont worry sir, if you can reshare the corrected salary slip now, we can move this forward quickly
Customer: See… again you're saying "don't worry." I *am* worried. I have sent the corrected slip. Check it properly this time, okay? Once you confirm it's correct, tell me clearly what happens next.
Salesperson: Sure sir, you have my number too. Lets move into action. Sorry for the delay in reporting the KYC incomplete status to you earlier.
Customer: Alright… at least you're acknowledging it. I've sent the corrected slip. Check it once and tell me clearly — is everything okay now, or is there anything else pending?
Salesperson: Sure sir, Thank you for your patience . I will keep you updated."""

    # Check if GROQ_API_KEY is set
    if not os.getenv("GROQ_API_KEY"):
        logger.error("ERROR: GROQ_API_KEY environment variable is not set!")
        logger.error("Please set it in your .env file or environment variables.")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("CONVERSATION TRANSCRIPT EVALUATION")
    logger.info("=" * 80)
    logger.info("")

    # Parse the transcript
    logger.info("Parsing conversation transcript...")
    transcript = parse_transcript(conversation_transcript_text)
    logger.info(f"Parsed {len(transcript)} messages from transcript")
    logger.info("")

    # Display the transcript
    logger.info("=" * 80)
    logger.info("CONVERSATION TRANSCRIPT")
    logger.info("=" * 80)
    for entry in transcript:
        logger.info(f"{entry['role']}: {entry['message']}")
    logger.info("")

    # Run evaluation
    logger.info("=" * 80)
    logger.info("GENERATING EVALUATION REPORT...")
    logger.info("=" * 80)
    logger.info("This may take a moment...")
    logger.info("")

    try:
        evaluation_report = await evaluate_conversation(transcript)

        # Display the evaluation report
        logger.info("")
        logger.info("=" * 80)
        logger.info("EVALUATION REPORT")
        logger.info("=" * 80)
        logger.info("")
        print(evaluation_report)
        logger.info("")
        logger.info("=" * 80)

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_file = f"evaluation_report_{timestamp}.txt"
        transcript_file = f"conversation_transcript_{timestamp}.txt"

        # Save transcript
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write("=== CONVERSATION TRANSCRIPT ===\n\n")
            for entry in transcript:
                f.write(f"[{entry['timestamp']}] {entry['role']}: {entry['message']}\n")

        # Save evaluation
        with open(evaluation_file, "w", encoding="utf-8") as f:
            f.write("=== EVALUATION REPORT ===\n\n")
            f.write(evaluation_report)

        logger.info(f"")
        logger.info(f"Files saved:")
        logger.info(f"  - Transcript: {transcript_file}")
        logger.info(f"  - Evaluation: {evaluation_file}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        logger.error(f"Error details: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
