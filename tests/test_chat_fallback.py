import unittest


class TestChatFallback(unittest.TestCase):
    def test_generator_greeting_without_model(self) -> None:
        from app.generator import QwenGenerator

        gen = QwenGenerator(model_id="dummy")
        gen._loaded = True
        gen._tokenizer = None
        gen._model = None

        text = gen.generate_chat_response(message="hey", response_language="en")
        self.assertTrue(text.strip())
        self.assertNotIn("user response", text.lower())
        self.assertNotIn("assistant,", text.lower())

    def test_clean_dialogue_artifacts_strips_training_pair_format(self) -> None:
        from app.generator import _clean_dialogue_artifacts

        raw = (
            "Assistant, i am a student who wants to know about the best universities for international students. "
            "can you help me? User response: I'm sorry, but I don't have any specific information on the best "
            "universities for international students. What would you like to know?"
        )
        cleaned = _clean_dialogue_artifacts(raw)
        self.assertTrue(cleaned.strip())
        self.assertTrue(cleaned.lower().startswith("i'm sorry"))
        self.assertNotIn("user response", cleaned.lower())
        self.assertNotIn("assistant,", cleaned.lower())

    def test_generator_smalltalk_name_without_model(self) -> None:
        from app.generator import QwenGenerator

        gen = QwenGenerator(model_id="dummy")
        gen._loaded = True
        gen._tokenizer = None
        gen._model = None

        text = gen.generate_chat_response(message="whats ur name", response_language="en")
        self.assertTrue(text.strip())
        self.assertIn("advisor", text.lower())

    def test_generator_smalltalk_weather_without_model(self) -> None:
        from app.generator import QwenGenerator

        gen = QwenGenerator(model_id="dummy")
        gen._loaded = True
        gen._tokenizer = None
        gen._model = None

        text = gen.generate_chat_response(message="how's the weather", response_language="en")
        self.assertTrue(text.strip())
        self.assertIn("weather", text.lower())

    def test_chatbot_no_profile_is_language_aware(self) -> None:
        from app.chatbot import answer_question
        from app.models import UserProfile

        profile = UserProfile(
            bac_stream="",
            expected_grade_band="",
            motivation="",
            budget_band="",
            city="",
            country="MA",
        )
        resp = answer_question(
            question="recommend me schools in casablanca",
            profile=profile,
            schools={},
            transcripts=[],
            top_k=5,
            mode="recommendation",
        )
        self.assertTrue(resp.short_answer.strip())
        self.assertNotIn("On peut faire mieux", resp.short_answer)


if __name__ == "__main__":
    unittest.main()
