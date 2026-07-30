from src.NER.model import NERmodel


def test_parse_output_supports_single_pipe_delimiter():
    output = "metformin | THUỐC | Đơn thuốc | None | Bệnh nhân dùng metformin 500mg mỗi ngày"

    spans = NERmodel.parse_output(output)

    assert len(spans) == 1
    assert spans[0].text == "metformin"
    assert spans[0].typ == "THUỐC"
    assert spans[0].section == "Đơn thuốc"
    assert spans[0].subsection == "None"
    assert spans[0].context == "Bệnh nhân dùng metformin 500mg mỗi ngày"
