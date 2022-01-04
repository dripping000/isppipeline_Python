'''
列表元组字典
'''
tuple = ( 'runoob', 786 , 2.23, 'john', 70.2 )
list = [ 'runoob', 786 , 2.23, 'john', 70.2 ]
#tuple[2] = 1000    # 元组中是非法应用
list[2] = 1000  # 列表中是合法应用
print(tuple)               # 输出完整元组
print(tuple[0])            # 输出元组的第一个元素
print(tuple[1:3])        # 输出第二个至第四个（不包含）的元素
print(tuple[2:])         # 输出从第三个开始至列表末尾的所有元素
print(list[0])           # 输出列表的第一个元素
print(list[1:3])          # 输出第二个至第三个元素
print(list[2:])        # 输出从第三个开始至列表末尾的所有元素

tuple_list=[(1,2),(3,4)]
tuple_list[0]=(1,2)     #合法
#tuple_list[0][0]=1     #非法

dict = {}
dict['one'] = "This is one"
dict[2] = "This is two"
tinydict = {'name': 'john', 'code': 6734, 'dept': 'sales'}
print(dict['one']) # 输出键为'one' 的值
print(dict[2])  # 输出键为 2 的值
print(tinydict)  # 输出完整的字典
print(tinydict.keys())  # 输出所有键
print(tinydict.values())  # 输出所有值