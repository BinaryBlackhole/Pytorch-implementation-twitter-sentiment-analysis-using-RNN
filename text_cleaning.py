"""
Author: Sagar Chakraborty
Updated: Modern implementation with type hints and improved practices
"""

import re
from typing import List, Union


def cleanup_text(texts: Union[List[str], str]) -> List[str]:
    """
    Clean and preprocess tweet text for sentiment analysis.

    This function performs the following cleaning operations:
    - Removes HTML entities (&quot;, &amp;)
    - Replaces emoticons with TOKEMOTICON token
    - Replaces URLs with TOKURL token
    - Replaces mentions (@username) with TOKMENTION token
    - Replaces hashtags (#topic) with TOKHASHTAG token
    - Replaces dollar amounts ($123) with TOKDOLLAR token
    - Removes punctuation
    - Removes extra whitespace and newlines
    - Converts to lowercase

    Args:
        texts: Single string or list of strings to clean

    Returns:
        List of cleaned text strings
    """
    # Handle single string input
    if isinstance(texts, str):
        texts = [texts]

    cleaned_text = []

    for text in texts:
        # Remove HTML entities
        text = re.sub(r"&quot;(.*?)&quot;", r"\g<1>", text)
        text = re.sub(r"&amp;", "", text)

        # Replace emoticons with a token
        # Matches various emoticon patterns like :), :(, <3, etc.
        emoticon_pattern = (
            r"(^| )(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|"
            r"[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)"
        )
        text = re.sub(emoticon_pattern, r"\g<1>TOKEMOTICON", text)

        # Convert to lowercase but preserve tokens
        text = text.lower()
        text = text.replace("tokemoticon", "TOKEMOTICON")

        # Replace URLs with token
        url_pattern = (
            r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))"
            r"([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
        )
        text = re.sub(url_pattern, "TOKURL", text)

        # Replace mentions with token
        text = re.sub(r"@[\w]+", "TOKMENTION", text)

        # Replace hashtags with token
        text = re.sub(r"#[\w]+", "TOKHASHTAG", text)

        # Replace dollar amounts with token
        text = re.sub(r"\$\d+(?:\.\d+)?", "TOKDOLLAR", text)

        # Remove punctuation (keep only alphanumeric and spaces)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Remove newlines
        text = re.sub(r"\n", " ", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        cleaned_text.append(text)

    return cleaned_text


def simple_tokenizer(text: str) -> List[str]:
    """
    Simple whitespace tokenizer for text.

    Args:
        text: Input text string

    Returns:
        List of tokens
    """
    return text.split()


# Example usage
if __name__ == "__main__":
    # Test the cleaning function
    sample_tweets = [
        "@user Hello! Check out this link: https://example.com #NLP :)",
        "I love this $100 deal &amp; it's great!",
        "What an amazing   product!! @company #awesome"
    ]

    print("Original tweets:")
    for tweet in sample_tweets:
        print(f"  {tweet}")

    print("\nCleaned tweets:")
    cleaned = cleanup_text(sample_tweets)
    for tweet in cleaned:
        print(f"  {tweet}")
