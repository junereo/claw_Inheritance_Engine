import json
from src.tools import parse_text_to_story_json, StoryJSON

test_input = """
# META
TITLE: 호텔 영감
GENRE: 오컬트 스릴러
SYNOPSIS: 작가들이 갇힌 호텔...

# PHASE 1
[NARRATION] 강원도 평창, 눈보라가 친다.
[IMAGE] 눈 덮인 오래된 호텔의 외관.
[DIALOGUE | 이수연] 환영합니다 작가님들.

# PHASE 1 CHOICE
QUESTION: 복도에서 기묘한 책을 발견했다.
IMAGE_DESCRIPTION: 책장 앞의 낡은 일기장
---
OPTION 1: 책을 집어 든다 | 진실에 접근합니다 | 진실을 발견한다 | [MAP: -1, 1, 0, 0]
OPTION 2: 지나친다 | 위험을 회피합니다 | 기억을 잃는다 | [MAP: 1, 0, 1, 1]

# ENDING POSTER
TITLE_KO: 피의 호텔
IMAGE_DESCRIPTION: 붉은 달이 뜬 호텔 포스터
FOOTER: 절대 잊지 마세요.

# ENDING LIST
- ID: ending-a | CONDITION: 진실을 찾음 | BADGE: TRUE_END | TAGLINE: 진실은 차갑다 | LINES: 눈이 내린다, 호텔은 조용하다, 당신은 탈출했다, 그러나...
- ID: ending-b | CONDITION: 기억을 잃음 | BADGE: BAD_END | TAGLINE: 영원한 투숙 | LINES: 또 다른 작가, 또 다른 호텔, 영원히 반복되는, 비극

# INTERFACE
BUTTONS: 다시 시작, 홈으로
BRAND_TEXT: PromToon Inheritance Engine
"""

def test_parser():
    print("Starting parser test...")
    payload = parse_text_to_story_json(test_input)
    print("\n--- Parsed Payload ---")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    
    print("\n--- Validating with Pydantic ---")
    try:
        story = StoryJSON(**payload)
        print("✅ Pydantic validation PASSED!")
        
        # Check specific fields
        assert story.meta.title == "호텔 영감"
        assert len(story.phases) == 1
        assert story.phases[0].phaseNumber == 1
        assert len(story.phases[0].scenes) == 3
        assert story.phases[0].choice.question == "복도에서 기묘한 책을 발견했다."
        assert len(story.phases[0].choice.choices) == 2
        assert story.ending.poster.titleKo == "피의 호텔"
        assert len(story.ending.endings) == 2
        assert story.ending.buttons == ["다시 시작", "홈으로"]
        print("✅ Content assertions PASSED!")
        
    except Exception as e:
        print(f"❌ Validation FAILED: {e}")
        exit(1)

if __name__ == "__main__":
    test_parser()
