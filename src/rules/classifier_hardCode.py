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
    "isHistorical": ["Tiền sử bệnh", 
                     "Tiền sử bệnh nội khoa",
                     "Tiền sử bệnh ngoại khoa",
                     "Tiền sử bệnh", "trước khi",
                     "Tiền sử phẫu / thủ thuật",
                     "Tiền sử phẫu thuật", "trước khi nhập viện",
                     "Tiền sử thủ thuật",
                     "đã thực hiện",],
    
    "isFamily": ["Di truyền", "Gia đình", "Tiền sử gia đình"],
    
    "isNegated": ["không", "chưa",
                  "không có", "chưa có",
                  "không ghi nhận", "không phát hiện",
                  "âm tính", "phủ nhận",
                  "loại trừ",]
}

type_cls = {
    "TRIỆU_CHỨNG": ["bệnh", "triệu chứng",
                    "yếu tố", "tình trạng"],
    
    "TÊN_XÉT_NGHIỆM": ["thủ thuật"],
    
    "KẾT_QUẢ_XÉT_NGHIỆM": ["kết quả xét nghiệm", "kết quả khám",
                           "kết quả thăm khám", "đánh giá tại bệnh viện",
                           "chỉ số"],
    
    "CHẨN_ĐOÁN": ["chẩn đoán", "đánh giá tại bệnh viện"],
    
    "THUỐC": ["thuốc"],
}

