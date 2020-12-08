#Module Name : emp_test.py
list = [[1,2,3,] , [4,5,6,]]
print(list)
# for l in list:
#     print(l, end="")
print(list[1][1])
      # list[0:10]
#----------------------------------------------
import numpy as np
arr = np.array(list)
print(arr)
print(arr[1,1])


# for i in range(1,11):

#------------ JAVA ----------------------------
# int[] arr = new int[] {1,'2',3,4,5}
# list  = new ArrayList()
# list.add(1)
# list.add("A")
#
# Java Casting -----------
# int a = 10
# double d;
# a = (int)d

# python Casting -----------
a = 10
b = 7.4
a = int(b)

arr = np.array([1, 2 ,10.5])
type(arr)
arr.astype('float64')
print(arr.dtype,  type(arr))

# arange(10)  #0 1 2 3 4 5 6 7 8 9
# arr[ :3]  # 0 1 2
# arr[3:]   # 3 4 5 6 7 8 9
# arr[ : ]  # 0 1 2 3 4 5 6 7 8 9

# arr[행s : 행e ]

# 2차원배열 == 행렬(matrix) == 데이터프레임
# arr[행s:행e ,  열s:열e ]

arr = np.arange(9) #0 1 2 3 4 5 6 7 8
print(arr)         #[0 1 2 3 4 5 6 7 8]
arr = arr.reshape(3,3)
print(arr)
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]

# arr[ : , : ]  #모두출력
arr[ : , : ]


list = [1,2]
print(list * 2) #[1, 2, 1, 2]

arr = np.array(list)
print(arr * 2)    #[2 4]

list1 = [1,2]
list2 = [3,4]
# print(list1 * list2)
print(list1 + list2)  #[1, 2, 3, 4]

arr1 = np.array(list1)
arr2 = np.array(list2)
print(arr1 * arr2)     #[3 8]
print(arr1 + arr2)     #[4 6]

list1 = [1,2]
list2 = [3,4,5]
arr1 = np.array(list1)
arr2 = np.array(list2)
# print(arr1 * arr2)    #error ~ with shapes (2, ) (3, )

#matrix array(2차) : 내적 np.dot()   전치:T
list1 = [[1,1],[2,2]]
list2 = [3,3]
# len(list2)  #2
arr1 = np.array(list1)
arr2 = np.array(list2)
print(arr1.dot(arr2) )   #np.dot(A,B)
#[4 6]
#[6 12]