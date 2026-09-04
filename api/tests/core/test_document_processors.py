from app.core.document_processors import (
    ExtractedTable,
    TableExtractor,
    ListExtractor,
    CodeBlockExtractor,
    HeaderExtractor,
    DocumentProcessor,
)


def test_extracted_table_to_text():
    table = ExtractedTable(
        headers=["Name", "Age"], rows=[["Alice", "30"], ["Bob", "25"]], caption="People"
    )
    text = table.to_text()
    assert "Table: People" in text
    assert "Columns: Name | Age" in text
    assert "Alice | 30" in text
    assert "Bob | 25" in text


def test_extracted_table_to_qa_pairs():
    table = ExtractedTable(
        headers=["Name", "Age", "City"],
        rows=[["Alice", "30", "NYC"], ["Bob", "25", "LA"]],
    )
    qa_pairs = table.to_qa_pairs()
    assert ("What is Name?", "Alice") in qa_pairs
    assert ("What is the Age of Alice?", "30") in qa_pairs
    assert ("What is the City of Alice?", "NYC") in qa_pairs
    assert ("What is Name?", "Bob") in qa_pairs
    assert ("What is the Age of Bob?", "25") in qa_pairs
    assert ("What is the City of Bob?", "LA") in qa_pairs


def test_extract_markdown_tables():
    extractor = TableExtractor()
    markdown = """
| Header 1 | Header 2 |
| -------- | -------- |
| Cell 1,1 | Cell 1,2 |
| Cell 2,1 | Cell 2,2 |
    """
    tables = extractor.extract_markdown_tables(markdown)
    assert len(tables) == 1
    assert tables[0].headers == ["Header 1", "Header 2"]
    assert tables[0].rows == [["Cell 1,1", "Cell 1,2"], ["Cell 2,1", "Cell 2,2"]]


def test_extract_html_tables():
    extractor = TableExtractor()
    html = """
    <table>
        <tr><th>Col1</th><th>Col2</th></tr>
        <tr><td>Data1</td><td>Data2</td></tr>
        <tr><td>Data3</td><td>Data4</td></tr>
    </table>
    """
    tables = extractor.extract_html_tables(html)
    assert len(tables) == 1
    assert tables[0].headers == ["Col1", "Col2"]
    assert tables[0].rows == [["Data1", "Data2"], ["Data3", "Data4"]]


def test_extract_bullet_lists():
    extractor = ListExtractor()
    text = """
- Item 1
- Item 2
* Item 3
    """
    lists = extractor.extract_bullet_lists(text)
    assert len(lists) == 1
    assert lists[0] == ["Item 1", "Item 2", "Item 3"]


def test_extract_numbered_lists():
    extractor = ListExtractor()
    text = """
1. First
2. Second
3) Third
    """
    lists = extractor.extract_numbered_lists(text)
    assert len(lists) == 1
    assert lists[0] == ["First", "Second", "Third"]


def test_extract_fenced_blocks():
    extractor = CodeBlockExtractor()
    text = """
```python
def foo():
    pass
```
```
echo "hello"
```
    """
    blocks = extractor.extract_fenced_blocks(text)
    assert len(blocks) == 2
    assert blocks[0] == ("python", "def foo():\n    pass")
    assert blocks[1] == ("text", 'echo "hello"')


def test_extract_headers():
    extractor = HeaderExtractor()
    text = """
# Title
## Subtitle
### Section
    """
    headers = extractor.extract_headers(text)
    assert len(headers) == 3
    assert headers[0][:2] == (1, "Title")
    assert headers[1][:2] == (2, "Subtitle")
    assert headers[2][:2] == (3, "Section")


def test_build_toc():
    extractor = HeaderExtractor()
    text = """
# Title
## Subtitle
    """
    toc = extractor.build_toc(text)
    assert len(toc) == 2
    assert toc[0]["level"] == 1
    assert toc[0]["text"] == "Title"
    assert "position" in toc[0]
    assert toc[1]["level"] == 2
    assert toc[1]["text"] == "Subtitle"
    assert "position" in toc[1]


def test_document_processor():
    processor = DocumentProcessor()
    text = """
# Main Document

Here is a list:
- Item A
- Item B

And a table:
| Key | Value |
| --- | ----- |
| K1  | V1    |

```python
print("test")
```
    """
    result = processor.process(text)

    assert len(result["headers"]) == 1
    assert len(result["toc"]) == 1
    assert len(result["bullet_lists"]) == 1
    assert len(result["tables"]) == 1
    assert len(result["code_blocks"]) == 1

    assert result["bullet_lists"][0] == ["Item A", "Item B"]
    assert result["code_blocks"][0] == {"language": "python", "code": 'print("test")'}


def test_document_processor_enhance_text():
    processor = DocumentProcessor()
    text = """
- Item A
- Item B

| Key | Value |
| --- | ----- |
| K1  | V1    |
    """
    enhanced = processor.enhance_text_for_retrieval(text)
    assert "Item A" in enhanced
    assert "Q: What is Key? A: K1" in enhanced
    assert "Q: What is the Value of K1? A: V1" in enhanced
    assert "List items: Item A, Item B" in enhanced
