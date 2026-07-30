"""
Classification Order:
    Assert -> Type -> Candidate
    Check the Subsection the text is in then Section
        Reasoning:
            "Tiền sử bệnh" can have Subsection "Kết quả xét nghiệm"
            "Tiền sử bệnh" can have Subsecton "Thuốc trước khi nhập viện"
"Tiền sử bệnh hiện tại" I don't know if it means historical or not
"""

# Common Phrases/Words
    # If subsection contains these words -> That class
    # If contains >=2 Classes or NO class -> AI Inference AND Retrieval from DB
    # If contains numbers -> Likely "KẾT_QUẢ_XÉT_NGHIỆM"
    
assert_cls = {
    "isHistorical": [
        # Section headers
        "tiền sử",
        "tiền sử bệnh",
        "tiền sử bệnh nội khoa",
        "tiền sử nội khoa",
        "tiền sử bệnh ngoại khoa",
        "tiền sử phẫu / thủ thuật",
        "tiền sử phẫu thuật",
        "tiền sử thủ thuật",
        "tiền căn",

        # Temporal cues
        "trước đây",
        "trước đó",
        "trước khi",
        "cách đây",
        "trong quá khứ",

        # Before admission
        "trước nhập viện",
        "trước khi nhập viện",
        "trước lúc nhập viện",
        "cách nhập viện",
        "các sự kiện trước khi nhập viện",

        # Past medical history
        "đã từng",
        "đã dùng",
        "đã được chẩn đoán",
        "đã thực hiện",

        # Chronic disease history
        "bệnh lý nội khoa mạn",
        "bệnh lý mạn tính",
        "các bệnh lý nội khoa",
        "bệnh mạn tính",
        "tập kinh lâm sàng trước",

        # Medication history
        "thuốc trước",
        "thuốc đang dùng",
        "tại nhà",
        "đang dùng tại nhà",
    ],
    
    "isFamily": [
        "di truyền",
        "tiền sử gia đình",
        "bệnh sử gia đình",
        "gia đình", "người nhà", "họ hàng",
        "bố", "mẹ", "cha", "ba", "má",
        "ông", "bà",
        "ông nội", "bà nội",
        "ông ngoại", "bà ngoại",
        "anh", "chị", "em",
        "anh trai", "chị gái", "em trai", "em gái",
        "con", "cậu", "dì", "chú", "bác", "cô",
        ],
    
    "isNegated": [
        "không có",
        "không thấy",
        "không ghi nhận",
        "không còn",
        "chưa ghi nhận",
        "chưa có",
        "loại trừ",
        "phủ định",
        "phủ nhận",
        "âm tính",
        "không bị",
        "không do",
        "không phát hiện",
        "không",
        "chưa",
        "ko",],
}
