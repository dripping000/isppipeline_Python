'''
判断循环
'''
flag = False
name = 'luren'
if name == 'python':         # 判断变量是否为 python
    flag = True              # 条件成立时设置标志为真
    print('welcome boss')     # 并输出欢迎信息
else:
    print(name)               # 条件不成立时输出变量名称

num = 5
if num == 3:            # 判断num的值
    print('boss')
elif num == 2:
    print('user')
elif num == 1:
    print('worker')
elif num < 0:           # 值小于零时输出
    print('error')
else:
    print('roadman')     # 条件均不成立时输出

nums = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [3, 4, 7]]
total = 0
for i in nums:
    for j in i:
        total += j
print(total)

def find_target(target, nums):#找出排序数组的索引
    for i in range(len(nums)):
        if nums[i]==target:
           return i
print(find_target(5, [1,3,5,6]))


count = 0
while (count < 9):
    print('The count is:', count)
    count = count + 1
    print("Good bye!")

count = 0
while count < 5:
    print(count, "is  less than 5")
    count = count + 1
else:
    print(count, 'is not less than 5')
