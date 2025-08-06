from openai import OpenAI
import time
from datetime import datetime

def classify_word_complexity(
    profile: dict,
    word: str,
    api_key: str,
    max_retries: int = 3,
    retry_delay: int = 5
) -> int:
    """
    Classify word complexity using DeepSeek API with your original prompt structure.
    
    Args:
        profile: User profile dictionary with:
            - native_language
            - age_group
            - education_level
            - job_category
            - english_level
            - reading_domain
        word: Word to classify
        api_key: DeepSeek API key
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        int: 0 if word is simple, 1 if complex (with 0 as fallback)
    """
    # Debug header
    print(f"\n{'='*30}\nClassifying: '{word}'")
    print(f"Profile: {profile['english_level']} English speaker, {profile['age_group']}, {profile['job_category']}")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        timeout=15
    )

    # Your original exact prompt
    prompt = f"""You are simulating a non-native English learner with the following profile:

- Age: {profile['age_group']} years old
- Native language: {profile['native_language']}
- Education level: {profile['education_level']}
- Job category: {profile['job_category']}
- English reading proficiency: {profile['english_level']}
- Primary reading domain: {profile['reading_domain']}

As such a learner, carefully assess the complexity of English words based on all these factors:
- Your native language may make some words easier or harder to understand depending on linguistic similarity.
- Your education and job may influence which types of words you're more familiar with.
- Your life stage affects your exposure to certain vocabulary.
- Your main reading domain may bias your vocabulary toward specific fields.
- Even advanced learners may struggle with rare, domain-specific, abstract, or culturally unfamiliar words.

Given all of this, decide whether the following word would be considered complex or simple **for you**.

- Respond with only **0** or **1**:
  - 0 = The word is simple/easy to understand.
  - 1 = The word is complex/difficult to understand.

Word: "{word}"

Please reply with only the digit 0 or 1."""

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            print(f"\nAttempt {attempt + 1}/{max_retries}")
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that classifies words as simple or complex for a user based on their profile. Respond with only 0 or 1."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=1,
                top_p=1,
                stream=False
            )
            
            # Debug info
            api_time = time.time() - start_time
            content = response.choices[0].message.content.strip()
            print(f"API Response ({api_time:.2f}s): '{content}'")
            
            # Parse response
            if content in ("0", "1"):
                result = int(content)
                print(f"✅ Final classification: {result} ({'simple' if result == 0 else 'complex'})")
                return result
            
            # Fallback parsing
            for token in content.split():
                if token in ("0", "1"):
                    print(f"⚠️ Extracted from response: {token}")
                    return int(token)
            
            print(f"❌ Unexpected response format, defaulting to simple")
            return 1

        except Exception as e:
            print(f"⚠️ Error: {str(e)}")
            if attempt == max_retries - 1:
                print(f"❌ All attempts failed, defaulting to simple")
                return 0
            
            print(f"Waiting {retry_delay} seconds before retry...")
            time.sleep(retry_delay)
    
    return 1 # Final fallback