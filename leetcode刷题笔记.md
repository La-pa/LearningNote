# leetcode刷题笔记
## 刷题计划
## 刷题内容
### 数组
#### 解题方法
* 双指针法
* 模拟
* 滑动窗口
* 二分查找
#### 想法
* 在双指针法中，要确定好定量，到底是左闭右开，还是左闭右闭，还有left <= right
* 排序是最后考虑的方法，因为排序实在是太慢了，要$O(n\log_{}{n})$时间
* 交换函数要进行三次赋值，如果只需要进行一次赋值就不要采用swap
* 二分查找的方法要熟练，尽量写成mid = left + (right - left )/2。因为后面便于插值查找的理解

### 链表
#### 解题方法
* 移除数组采用单指针法
* 注意指针所指向的空间是否为null
* 双指针法
* 反转链表
* 虚拟头指针：就是在头指针前面多new一个空间来指向头指针，new的空间称为虚拟指针，这样在处理问题的时候就不必重复考虑头指针的情况，因为头指针和其他节点一样都作为虚拟指针的节点，返回时要返回虚拟指针的next
* 快慢指针，fast每次移动2个节点，slow每次移动1个节点


### 哈希表
#### set && map
* set

| 集合               | 底层实现 | 是否有序 | 数值是否可以重复 | 是否可以更改数值 | 查询效率 | 增删效率 |
| :----------------- | -------- | -------- | ---------------- | ---------------- | -------- | -------- |
| std::set           | 红黑树   | 有序     | 否               | 否               | O(log n) | O(log n) |
| set::multiset      | 红黑树   | 有序     | 是               | 否               | O(log n) | O(log n) |
| std::unordered_set | 哈希表   | 无序     | 否               | 否               | O(1)     | O(1)     |

* map

| 映射               | 底层实现 | 是否有序 | 数值是否可以重复 | 是否可以更改数值 | 查询效率 | 增删效率 |
| :----------------- | -------- | -------- | ---------------- | ---------------- | -------- | -------- |
| std::map           | 红黑树   | key有序  | key不可重复      | key不可修改      | O(log n) | O(log n) |
| set::multimap      | 红黑树   | key有序  | key不可重复      | key不可修改      | O(log n) | O(log n) |
| std::unordered_map | 哈希表   | key无序  | key不可重复      | key不可修改      | O(1)     | O(1)     |



* 使用时要包括头文件set和unordered_set以及multiset属于不同的头文件

#### 适用时机
* 当我们遇到了要快速判断一个元素是否出现集合里的时候，就要考虑哈希法了。
* 只有对这些数据结构的底层实现很熟悉，才能灵活使用，否则很容易写出效率低下的程序。
#### 解题方法
* 通过数组来判断字符出现的次数，字符下标代表ascii的顺序
* 对于小规模的数据用数组比较合算，速度更快
* 对于大规模的数据用哈希表比较合算
* 而unordered_set 和 unordered_map的底层是哈希表可以简化操作
##### set作为哈希表
###### 使用场景
* 题目中我们给出了什么时候用数组就不行了，需要用set。如果这道题目没有限制数值的大小，就无法使用数组来做哈希表了。
* 主要因为如下两点：
> 数组的大小是有限的，受到系统栈空间（不是数据结构的栈）的限制。
> 如果数组空间够大，但哈希值比较少、特别分散、跨度非常大，使用数组就造成空间的极大浪费。

```C++
所以此时一样的做映射的话，就可以使用set了。

关于set，C++ 给提供了如下三种可用的数据结构：
std::set
std::multiset
std::unordered_set
std::set和std::multiset底层实现都是红黑树，std::unordered_set的底层实现是哈希， 使用unordered_set 读写效率是最高的。

```
##### map作为哈希表

###### 使用数组和set来做哈希法的局限。

* 数组的大小是受限制的，而且如果元素很少，而哈希值太大会造成内存空间的浪费。
* set是一个集合，里面放的元素只能是一个key，而两数之和这道题目，不仅要判断y是否存在而且还要记录y的下标位置，因为要返回x 和 y的下标。所以set 也不能用。
* map是一种<key, value>的结构，可以用key保存数值，用value在保存数值所在的下标。所以使用map最为合适。

###### map介绍
C++提供如下三种map：：
* std::map
* std::multimap
* std::unordered_map
> std::unordered_map 底层实现为哈希，std::map 和std::multimap 的底层实现是红黑树。
>同理，std::map 和std::multimap 的key也是有序的（这个问题也经常作为面试题，考察对语言容器底层的理解）
### 字符串
#### 解题方法
##### 双指针法
* 优先采用双指针法，时间复杂度小
* 但是代码比较复杂

##### KMP


##### 反转系列
* 可以使用reserve函数进行字符串的反转
* 同时反转比较考察对代码的把握能力
* 遇到一些题目可以采用局部反转+整体反转的方式
* 要熟练掌握字符串反转的基本功，不要太依赖库函数
* 同时用反转的话时间复杂度会比较大一些，但是空间复杂度比较小



### 二叉树

#### 经验

* 二叉树题目经常运用后序遍历来进行做答
* 因为有时候用递归法来做题, 代码量是比较小的
* 而递归法需要对之前递归的返回值进行判断
* 而后序遍历就是把打印的操作改成对数据处理的操作

## 解题方法

### 递归法

1. 确定递归函数的参数和返回值
2. 确定终止条件
3. 确定单层递归的逻辑





