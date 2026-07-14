import os
import json
import re

RESEARCH_SYSTEM_PROMPT = """
You are a senior research scientist conducting a literature review and analysis.
- Provide thorough, technically accurate responses
- Cite specific details, numbers, and benchmarks where relevant
- Use clear section headers and structured formatting
- Be precise about claims — distinguish between established facts and speculative trends
"""

PLAN_PROMPT = """
Decompose the following research query into 3-6 specific sub-questions that,
when answered together, provide a complete answer. For each sub-question,
specify whether it requires:
- "web_search" (needs current information)
- "code_exec" (needs code/verification)
- "both"
- "reasoning" (can be answered by reasoning alone)

Respond ONLY with valid JSON in this exact schema:
{
  "query": "<original query>",
  "sub_tasks": [
    {
      "id": 1,
      "question": "<specific question>",
      "tool": "web_search | code_exec | both | reasoning",
      "rationale": "<why this question matters>"
    }
  ]
}
"""

SYNTHESIS_PROMPT = """
You are writing a research report. You have findings from multiple sub-tasks.
Synthesize them into a well-structured markdown report with:
1. A title and brief overview
2. Sections for each major finding (use ## headers)
3. Specific data points, benchmarks, and citations where available
4. A conclusion with key takeaways
5. Sources section at the end

Be precise and technical. Avoid filler.
Respond ONLY with the markdown report content — no extra commentary.
"""

REVIEW_PROMPT = """
Review this research report for completeness and accuracy.
Check for:
1. Missing sections or unanswered sub-questions
2. Vague claims that need specific data
3. Factual errors or contradictions
4. Lack of citations or sources

Respond ONLY with valid JSON:
{
  "pass": true/false,
  "gaps": ["gap1", "gap2"],
  "feedback": "brief overall assessment",
  "missing_sub_task_ids": [1, 3]
}
"""


class ResearchLLM:
    def __init__(self, api_key=None, model_name="llama-3.1-8b-instant"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set GROQ_API_KEY environment variable."
            )
        self.model_name = model_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def _call(self, system, user, temperature=0.3, max_tokens=4096):
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system.strip()},
                {"role": "user", "content": user.strip()},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def generate_text(self, system, user, temperature=0.3, max_tokens=4096):
        return self._call(system, user, temperature, max_tokens)

    def generate_json(self, system, user, temperature=0.2, max_tokens=2048):
        raw = self._call(system, user, temperature, max_tokens)
        parsed = self._parse_json(raw)
        if parsed:
            return parsed
        return {"error": "Failed to parse LLM response", "raw": raw[:500]}

    def generate_plan(self, query):
        result = self.generate_json(PLAN_PROMPT, f"Query: {query}")
        result["original_query"] = query
        return result

    def synthesize_report(self, findings_text):
        report = self.generate_text(SYNTHESIS_PROMPT, findings_text, temperature=0.4, max_tokens=8192)
        return report

    def review_report(self, report_text, original_plan):
        user = f"Original plan:\n{json.dumps(original_plan, indent=2)}\n\nReport:\n{report_text}"
        return self.generate_json(REVIEW_PROMPT, user, temperature=0.1)

    def _parse_json(self, raw):
        cleaned = self._strip_markdown(raw)
        result = self._try_parse(cleaned)
        if result:
            return result
        result = self._try_parse(raw)
        if result:
            return result
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            result = self._try_parse(json_match.group(0))
            if result:
                return result
        return None

    def _strip_markdown(self, text):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?\s*```$", "", text)
        return text.strip()

    def _try_parse(self, text):
        if not text or not isinstance(text, str):
            return None
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
