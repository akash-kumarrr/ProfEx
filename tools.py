from langchain_core.tools import tool
import wikipedia
import arxiv

@tool("WikiSearch")
def WikiSearch(arg: str) -> str:
    """
    Tool name: WikiSearch
    Description: This tool parses Wikipedia search results for a particular input or search item.
    """
    try:
        results = wikipedia.search(arg, results=3)
        if not results:
            return "No Wikipedia results found."
        
        content = []
        for result in results[:3]:
            try:
                page = wikipedia.page(result)
                content.append(f"Title: {page.title}\n{page.content[:1000]}")
            except:
                continue
        return "\n---\n".join(content)
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

@tool("ArxivSearch")
def ArxivSearch(arg: str) -> str:
    """
    Tool name: ArxivSearch
    Description: This tool searches research papers on Arxiv for a particular topic.
    """
    try:
        client = arxiv.Client()
        results = client.results(
            arxiv.Search(query=arg, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
        )
        
        papers = []
        for result in results:
            papers.append(f"Title: {result.title}\nSummary: {result.summary[:1000]}")
        
        return "\n---\n".join(papers) if papers else "No papers found."
    except Exception as e:
        return f"Error searching Arxiv: {str(e)}"