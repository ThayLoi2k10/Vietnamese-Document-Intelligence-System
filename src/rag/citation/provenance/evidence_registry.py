"""Evidence registry mapping prompt reference identifiers to source context items."""

from typing import Dict, List, Optional

from src.rag.context.domain.models import ContextItem


class EvidenceRegistry:
    """Maintains indexed legal evidence items available in prompt context."""

    def __init__(self, items: Optional[List[ContextItem]] = None) -> None:
        self._items_by_id: Dict[str, ContextItem] = {}
        self._items_by_num: Dict[int, ContextItem] = {}
        if items:
            for it in items:
                self.register(it)

    def register(self, item: ContextItem) -> None:
        ref_id = item.context_item_id
        self._items_by_id[ref_id] = item

        bare_id = ref_id.strip("[]")
        self._items_by_id[bare_id] = item

        if bare_id.isdigit():
            self._items_by_num[int(bare_id)] = item
        elif item.context_position:
            self._items_by_num[item.context_position] = item

    def lookup_by_number(self, citation_number: int) -> Optional[ContextItem]:
        return self._items_by_num.get(citation_number)

    def lookup_by_id(self, reference_id: str) -> Optional[ContextItem]:
        return self._items_by_id.get(reference_id) or self._items_by_id.get(
            reference_id.strip("[]")
        )

    def get_all_items(self) -> List[ContextItem]:
        return list({id(it): it for it in self._items_by_id.values()}.values())

    def __len__(self) -> int:
        return len(self.get_all_items())
