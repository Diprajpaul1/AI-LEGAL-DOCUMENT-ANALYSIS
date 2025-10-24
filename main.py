import os
import time
import re
from dotenv import load_dotenv
from crew import legal_assistant_crew
from litellm.exceptions import RateLimitError

# ✅ Load environment file safely
ENV_PATH = os.path.join(os.path.dirname(__file__), "env_template.txt")
if not os.path.exists(ENV_PATH):
    raise FileNotFoundError(f"❌ Env file not found at: {ENV_PATH}")

load_dotenv(dotenv_path=ENV_PATH)

def run(user_input: str):
    max_retries = 5
    delay = 30  # initial delay in seconds (exponential backoff base)

    for attempt in range(1, max_retries + 1):
        try:
            result = legal_assistant_crew.kickoff(inputs={"user_input": user_input})
            return result

        except RateLimitError as e:
            msg = str(e)
            print(f"⚠️ Rate limit hit (attempt {attempt}/{max_retries}): {msg}")

            # Try to extract suggested wait time from Groq message
            wait_match = re.search(r"try again in (\d+)m(\d+\.?\d*)s", msg)
            if wait_match:
                minutes, seconds = map(float, wait_match.groups())
                total_wait = minutes * 60 + seconds
                print(f"⏳ Waiting {int(total_wait)} seconds before retry (per Groq)...")
                time.sleep(total_wait + 5)
            else:
                print(f"⏳ Waiting {delay}s before retry...")
                time.sleep(delay)
                delay *= 2  # exponential backoff

        except KeyError as e:
            print(f"❌ Missing environment variable: {e}")
            print("💡 Please make sure your env file includes all required keys.")
            break

        except Exception as e:
            print(f"🚨 Unexpected error: {type(e).__name__}: {e}")
            break

    print("❌ Failed after several retries — please wait a while and rerun.")

if __name__ == "__main__":
    user_input = (
        "A man broke into my house at night while my family was sleeping. "
        "He stole jewelry and cash from our bedroom. When I confronted him, "
        "he threatened me with a knife and ran away. We reported it to the police, "
        "but I'm not sure which legal charges should be filed under IPC."
    )
    result = run(user_input)
    if result:
        print("\n✅ Result:\n", result)
