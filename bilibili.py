from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader
# loader = WebBaseLoader("https://zh.wikipedia.org/wiki/%E8%A5%84%E9%98%B3%E5%B8%82")
# loader = UnstructuredHTMLLoader("https://www.hubei.gov.cn/")
#file_path = "./chuyiAI.md"
#loader = UnstructuredMarkdownLoader(file_path)
loader = UnstructuredImageLoader("./liang.jpg")
data = loader.load()
print(data)
