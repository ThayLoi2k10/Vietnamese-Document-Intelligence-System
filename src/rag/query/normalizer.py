"""Deterministic legal acronym normalizer for Vietnamese queries."""

from enum import Enum
import re
import unicodedata
from typing import Dict, List, Optional, Tuple


class NormalizationMode(str, Enum):
    EXPAND = "EXPAND"
    REPLACE = "REPLACE"
    DUAL_SPACE = "DUAL_SPACE"


LEGAL_ACRONYMS: Dict[str, str] = {
    "NLĐ": "người lao động",
    "NLD": "người lao động",
    "NSDLĐ": "người sử dụng lao động",
    "NSDLD": "người sử dụng lao động",
    "BHXH": "bảo hiểm xã hội",
    "BHYT": "bảo hiểm y tế",
    "BHTN": "bảo hiểm thất nghiệp",
    "HĐLĐ": "hợp đồng lao động",
    "HDLD": "hợp đồng lao động",
    "TNLĐ": "tai nạn lao động",
    "TNLD": "tai nạn lao động",
    "BNN": "bệnh nghề nghiệp",
    "ATVSLĐ": "an toàn vệ sinh lao động",
    "ATVSLD": "an toàn vệ sinh lao động",
    "ATLĐ": "an toàn lao động",
    "ATLD": "an toàn lao động",
    "TNCN": "thuế thu nhập cá nhân",
    "TNDN": "thuế thu nhập doanh nghiệp",
    "GTGT": "giá trị gia tăng",
    "TTĐB": "tiêu thụ đặc biệt",
    "TTDB": "tiêu thụ đặc biệt",
    "XNK": "xuất nhập khẩu",
    "TCTD": "tổ chức tín dụng",
    "NHNN": "ngân hàng nhà nước",
    "NHTW": "ngân hàng trung ương",
    "ĐKKD": "đăng ký kinh doanh",
    "DKKD": "đăng ký kinh doanh",
    "BĐS": "bất động sản",
    "BDS": "bất động sản",
    "QSDĐ": "quyền sử dụng đất",
    "QSDD": "quyền sử dụng đất",
    "GCNQSDĐ": "giấy chứng nhận quyền sử dụng đất",
    "GCNQSDD": "giấy chứng nhận quyền sử dụng đất",
    "GCN": "giấy chứng nhận",
    "VPĐKĐĐ": "văn phòng đăng ký đất đai",
    "VPDKDD": "văn phòng đăng ký đất đai",
    "VPHC": "vi phạm hành chính",
    "XLVPHC": "xử lý vi phạm hành chính",
    "PCCC": "phòng cháy chữa cháy",
    "CSGT": "cảnh sát giao thông",
    "ATGT": "an toàn giao thông",
    "TNGT": "tai nạn giao thông",
    "TTATGT": "trật tự an toàn giao thông",
    "CAND": "công an nhân dân",
    "QĐND": "quân đội nhân dân",
    "QDND": "quân đội nhân dân",
    "THADS": "thi hành án dân sự",
    "TAND": "tòa án nhân dân",
    "VKSND": "viện kiểm sát nhân dân",
    "UBND": "ủy ban nhân dân",
    "HĐND": "hội đồng nhân dân",
    "HDND": "hội đồng nhân dân",
    "BCA": "bộ công an",
    "BQP": "bộ quốc phòng",
    "BTC": "bộ tài chính",
    "BYT": "bộ y tế",
    "BTP": "bộ tư pháp",
    "BCT": "bộ công thương",
    "BGDĐT": "bộ giáo dục và đào tạo",
    "BGDDT": "bộ giáo dục và đào tạo",
    "BTNMT": "bộ tài nguyên và môi trường",
    "BLĐTBXH": "bộ lao động thương binh và xã hội",
    "BLDTBXH": "bộ lao động thương binh và xã hội",
    "TTCP": "thanh tra chính phủ",
    "TNHH": "trách nhiệm hữu hạn",
    "CP": "cổ phần",
    "HĐQT": "hội đồng quản trị",
    "HDQT": "hội đồng quản trị",
    "HĐMB": "hợp đồng mua bán",
    "HDMB": "hợp đồng mua bán",
    "VBQPPL": "văn bản quy phạm pháp luật",
    "QPPL": "quy phạm pháp luật",
    "VBHN": "văn bản hợp nhất",
}

VI_CHARS = (
    r"a-zA-Z0-9"
    r"àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
)


class LegalAcronymNormalizer:
    """Normalizes and expands legal acronyms in Vietnamese queries."""

    def __init__(
        self,
        custom_dict: Optional[Dict[str, str]] = None,
        mode: NormalizationMode = NormalizationMode.EXPAND,
    ) -> None:
        self.mode = mode
        self.acronym_dict = dict(LEGAL_ACRONYMS)
        if custom_dict:
            self.acronym_dict.update(custom_dict)

        self.sorted_acronyms = sorted(
            self.acronym_dict.keys(), key=lambda x: len(x), reverse=True
        )

        self.patterns = {
            acr: re.compile(
                rf"(?<![{VI_CHARS}]){re.escape(acr)}(?![{VI_CHARS}])",
                flags=re.IGNORECASE,
            )
            for acr in self.sorted_acronyms
        }

    def normalize(self, query: str) -> Tuple[str, Dict[str, str], List[str]]:
        if not query or not query.strip():
            return query, {}, []

        normalized_text = unicodedata.normalize("NFC", query.strip())
        acronyms_detected: Dict[str, str] = {}
        rules_applied: List[str] = []

        for acr in self.sorted_acronyms:
            pattern = self.patterns[acr]
            matches = pattern.findall(normalized_text)
            if matches:
                expansion = self.acronym_dict[acr]
                acronyms_detected[acr] = expansion

                def _replace(match: re.Match) -> str:
                    orig_matched = match.group(0)
                    if self.mode == NormalizationMode.REPLACE:
                        return expansion
                    elif self.mode == NormalizationMode.DUAL_SPACE:
                        return f"{orig_matched} {expansion}"
                    return f"{orig_matched} ({expansion})"

                normalized_text = pattern.sub(_replace, normalized_text)
                rules_applied.append(f"EXPAND_ACRONYM:{acr}->{expansion}")

        normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
        return normalized_text, acronyms_detected, rules_applied

    def get_supported_acronyms(self) -> Dict[str, str]:
        return dict(self.acronym_dict)
