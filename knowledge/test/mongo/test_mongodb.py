from pymongo import MongoClient

# 创建连接MongoDB对象
# # "mongodb://admin:123456@192.168.200.130:27017"
mongo_client = MongoClient("mongodb://192.168.139.130:27017")
# 数据库
db = mongo_client["know_db"]
# 集合（表）
collection = db["users"]

##################删除操作#################################
# 删除
def delete_data():
    result = collection.delete_one({"name": "王五"})

##################修改操作#################################
# update users set age=100 where name="张三"
def update_data():
    result = collection.update_one(
        {"name": "张三"},
        {"$set": {"age": 100}},
    )
    print(result)


##################查询操作#################################
# 对MongoDB集合文档 curd操作
def find_one():
    document = collection.find_one()

# 获取集合年龄最大的人员姓名
# select name from users order by age desc  limit  1
# 1 升序  -1 降序
def find_max_age():
    for document in (collection.find()
            .sort("age",-1).limit(1)):
        print(document['name'])

# 查询所有
def find_condional():
    for document in collection.find({"name":"李四"}):
        print(document)

# 查询所有
def find_all():
    for document in collection.find():
        print(document['name'])


##################添加操作#################################
# 1 添加一条记录
def insert_data():
    result = collection.insert_one(
        {
            "name": "张三",
            "age": 20,
            "major": "计算机科学"
        }
    )
    print(result)

# 2 添加多条记录
def insert_many_data():
    result = collection.insert_many(
        [
            {"name": "李四", "age": 22, "major": "软件工程"},
            {"name": "王五", "age": 21, "major": "计算机科学"},
        ]
    )
    print(result)

if __name__ == '__main__':
    # insert_data()
    # insert_many_data()
    # find_all()
    # find_condional()
    # find_max_age()
    # update_data()
    delete_data()