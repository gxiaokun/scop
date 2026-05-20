from pydantic import BaseModel, Field, field_validator
from typing import List, Tuple, Optional, Any
import json
from openai.types.chat import ChatCompletionMessageParam

from src.config.llm_config import LLMConfig
from src.llm.fewshots import ENTITY_ALIGN_FEW_SHOTS
from src.utils.llm_utils import LLMTaskExecutor

Triple = Tuple[str, str, str]


class GroundedTriplesSchema(BaseModel):

    anchor_triples: List[Triple] = Field(
        default_factory=list,
        description=(
            "Aligned anchor triples. Preserve the same number and order as "
            "source_triples.anchor_triples. Each triple is [head, relation, tail]."
        ),
    )

    search_triple: Optional[Triple] = Field(
        default=None,
        description=(
            "Aligned search triple. If the input search_triple is null, output null. "
            "If it exists, preserve it as exactly one triple and keep any '?' unchanged."
        ),
    )


class AlignSchema(BaseModel):
    grounded_triples: GroundedTriplesSchema = Field(
        description="Aligned anchor_triples and search_triple."
    )


ALIGN_SYSTEM_PROMPT = """
You are an entity and relation normalization system for TKGQA retrieval and graph alignment.

Input:
1. original_question:
   The original natural language question. Use it only for semantic disambiguation, relation-action judgment, and role/title coreference judgment.

2. gold:
   - triples: standard gold triples, each in [head, relation, tail] format.
   - entities: standard entity set.
   - relations: standard relation set.

3. source_triples:
   - anchor_triples: extracted anchor triples.
   - search_triple: extracted retrieval triple, which may be null.

Task:
Align source_triples to gold-standard entity and relation names for downstream graph retrieval.
This is not open-ended rewriting. Do not add facts. Do not answer the question.

Core principles:
1. Preserve structure.
2. Preserve triple direction.
3. Preserve "?" exactly.
4. Prefer conservative alignment.
5. Prefer original text when uncertain.
6. Use original_question only for disambiguation, not for adding new facts.

Output requirements:
1. The number and order of anchor_triples must be exactly the same as the input.
2. If input search_triple is null, output search_triple must be null.
3. If input search_triple exists, output exactly one search_triple.
4. Every triple must be [head, relation, tail].
5. If a slot is "?", keep it as "?" and do not fill, move, or replace it.
6. Do not swap head and tail. Do not change relation direction.
7. Each output slot must be either:
   - the original source text,
   - "?",
   - or a selected standard item from gold.entities / gold.relations / gold.triples.
   Do not invent a new normalized name.

Anchor alignment:
For each anchor triple, use the following priority order:

1. Complete gold-triple match:
   If the source anchor triple exactly or nearly matches a triple in gold.triples across head, relation, and tail, output that gold triple.

2. High-confidence complete-triple match:
   If one gold triple is clearly the best match considering entity type, relation action, country/region modifier, direction, and original_question, output that gold triple.

3. Conservative slot-level alignment:
   If no reliable complete gold triple exists:
   - align head/tail only to gold.entities when the match is reliable;
   - align relation only to gold.relations when the action type is reliable;
   - otherwise preserve the original slot text.

Do not force an anchor into a gold triple merely because one entity name is similar.
Do not create an arbitrary hybrid triple by combining slots from unrelated gold triples just to make the result look standardized.

Search-triple alignment:
For search_triple, preserve the query structure.

1. If search_triple is null, keep it null.
2. If it contains "?", keep "?" in the same slot.
3. Do not fill "?" using gold entities.
4. Do not change head/tail positions.
5. Do not change relation direction.
6. For non-"?" head/tail slots:
   - align to gold.entities only when the entity match is reliable;
   - otherwise preserve the original text.
7. For relation:
   - align to the gold.relations item with the closest matching action type;
   - otherwise preserve the original relation text.
8. gold.triples may help disambiguate search_triple, but do not rewrite search_triple as a complete gold triple unless it is already nearly identical.

Entity alignment:
1. Entity type consistency has priority.
2. Do not align different entity types just because names overlap.
   Examples: country ≠ government ≠ department ≠ president ≠ prime minister ≠ citizen ≠ police.
3. Prefer the candidate with the most consistent type, country/region modifier, context, and triple structure.
4. Do not infer a more specific entity unless original_question clearly supports it.
5. If no reliable entity match exists, preserve the original entity text.

Relation alignment:
1. Relation action type has priority over topical similarity.
2. Preserve event direction.
3. Intent / plan / wish / request relations must not be aligned to actually occurred actions.
4. Actually occurred actions must not be aligned to intent relations.
5. If no reliable relation match exists, preserve the original relation text.

Common mappings:
- "negotiate with" -> "Engage in negotiation"
- "want to meet with" / "intend to negotiate" -> "Express intent to meet or negotiate"
- "criticise" / "criticize" -> "Criticize or denounce"
- "praise" -> "Praise or endorse"
- "request" -> "Make an appeal or request"

Common confusions to avoid:
- occurred action ≠ expressed intent
- visit ≠ negotiate
- negotiate ≠ express intent to meet or negotiate
- criticize ≠ make a statement
- government ≠ country
- department ≠ country

Use of original_question:
Use original_question only to:
1. determine action semantics;
2. resolve role/title/institution references;
3. resolve coreference among mentions in the same input;
4. disambiguate entity or relation alignment.

Do not use original_question to add missing entities, add missing relations, or fabricate facts.

Final check before output:
1. anchor_triples count and order are unchanged.
2. search_triple null/non-null status is unchanged.
3. "?" remains in the original slot.
4. head/tail direction is unchanged.
5. Every aligned item is either selected from gold or preserved from the source.
6. Uncertain cases preserve the original text.
"""

def run_entity_align(
    source_triples: dict,
    gold_entities: List[str],
    gold_relations: List[str],
    gold_triples: List[List[str]],
    original_question: str,
) -> AlignSchema:
    
    llm_executor = LLMTaskExecutor()
    
    input_data = {
        "original_question": original_question,
        "gold": {
            "triples": gold_triples,
            "entities": gold_entities,
            "relations": gold_relations,
        },
        "source_triples": {
            "anchor_triples": source_triples.get("anchor_triples", []),
            "search_triple": source_triples.get("search_triple", None),
        },
    }

    response = llm_executor.execute_structured_task(
        system_prompt=ALIGN_SYSTEM_PROMPT,
        few_shots=ENTITY_ALIGN_FEW_SHOTS,
        current_input_data=input_data,
        response_model=AlignSchema
    )


    return response
