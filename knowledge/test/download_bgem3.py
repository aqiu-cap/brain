from modelscope import snapshot_download

str = snapshot_download(
    model_id="BAAI/bge-m3",
    local_dir="D:/know/bgem3"
)
print(str)


# from modelscope import snapshot_download
#
# str = snapshot_download(model_id="BAAI/bge-reranker-large",
#                         local_dir="D:/know/reranker")
# print(str)