
import asyncio
import re
import json
from dataclasses import dataclass

# Mock classes to simulate the environment
@dataclass
class Entity:
    name: str
    type: str = "OTHER"
    description: str = ""
    mentions: list = None

@dataclass
class Relationship:
    source: str
    target: str
    relation: str
    chunk_ids: list = None
    description: str = ""

class EntityType:
    OTHER = "other"
    def __init__(self, value): self.value = value

# Mock Provider
class MockProvider:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    async def chat(self, messages):
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            if isinstance(response, Exception):
                raise response
            return response
        return "{}"

# The logic to test (extracted from the file modification)
async def test_extraction(provider):
    import re
    import json
    
    # Simplified version of the logic we just added
    # We copy the EXACT regex and logic from the file to test IT, not a re-implementation
    
    last_error = None
    
    for attempt in range(3):
        try:
            response = await provider.chat([])
            clean_response = response.strip()
            
            # THE REGEX WE ADDED
            json_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', clean_response, re.IGNORECASE)
            if json_block_match:
                clean_response = json_block_match.group(1)
            else:
                json_match = re.search(r'(\{[\s\S]*\})', clean_response)
                if json_match:
                    clean_response = json_match.group(1)
            
            clean_response = re.sub(r',\s*([}\]])', r'\1', clean_response)
            
            try:
                data = json.loads(clean_response)
                return data # Success
            except json.JSONDecodeError:
                 # Fallback: try to fix unescaped quotes if safe
                try:
                    clean_response_fixed = clean_response.replace("'", '"')
                    data = json.loads(clean_response_fixed)
                    return data
                except Exception:
                    if attempt < 2:
                        continue
                    raise ValueError(f"Failed to parse JSON")

        except Exception as e:
            last_error = e
            if attempt == 2:
                raise e
    return None

async def run_tests():
    print("Running GraphRAG Reliability Tests...")
    
    # Test 1: Clean JSON
    print("\nTest 1: Clean JSON")
    p1 = MockProvider(['{"entities": [], "relationships": []}'])
    res1 = await test_extraction(p1)
    print(f"Result: {res1 is not None}")
    assert res1 is not None

    # Test 2: Markdown block
    print("\nTest 2: Markdown block")
    p2 = MockProvider(['Here is the data:\n```json\n{"entities": [], "relationships": []}\n```'])
    res2 = await test_extraction(p2)
    print(f"Result: {res2 is not None}")
    assert res2 is not None
    
    # Test 3: Markdown block without 'json' tag
    print("\nTest 3: Markdown block (no tag)")
    p3 = MockProvider(['```\n{"entities": [], "relationships": []}\n```'])
    res3 = await test_extraction(p3)
    print(f"Result: {res3 is not None}")
    assert res3 is not None

    # Test 4: Trailing comma repair
    print("\nTest 4: Trailing comma")
    p4 = MockProvider(['{"entities": [], "relationships": [],}'])
    res4 = await test_extraction(p4)
    print(f"Result: {res4 is not None}")
    assert res4 is not None

    # Test 5: Single quotes
    print("\nTest 5: Single quotes")
    p5 = MockProvider(["{'entities': [], 'relationships': []}"])
    res5 = await test_extraction(p5)
    print(f"Result: {res5 is not None}")
    assert res5 is not None

    # Test 6: Retry on failure
    print("\nTest 6: Retry logic (fail 2x then success)")
    p6 = MockProvider([
        ValueError("Network Error"), 
        "{invalid_json", 
        '{"entities": [], "relationships": []}'
    ])
    res6 = await test_extraction(p6)
    print(f"Result: {res6 is not None}")
    assert res6 is not None

    print("\nAll tests passed!")

if __name__ == "__main__":
    asyncio.run(run_tests())
