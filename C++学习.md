# C++学习
## 问题
* mutable
* struct 也是能有构造函数
* 左值
* 静态变量的用途
* 程序的四个区域
* 构造函数可以发生重载
* 析构函数不可以发生重载
* class和struct的区别

* 调用继承中重名的成员函数
* 继承中子类会继承父类中成员变量的值吗
* 继承中子类调用父类的构造函数
* 深拷贝和浅拷贝的原理
* new和malloc的区别
* new会调用对象的构造函数
* this
* 函数重载
* 友元
* 多态条件
* 有继承关系
* 子类重写父类的虚函数
* 父类的引用或指针指向子类对象**？？？**

* 类中成员函数存放在哪里

* 什么时候函数参数要传引用
* 函数返回多个值
* 函数返回引用

* 指针都是4个字节
* 虚函数指针vfptr：virtual funciton pointer
* 虚函数表
* 虚函数指针指向虚函数表：表内记录虚函数的入口地址




## 解决的问题

###  指针
* 指针常量：不可以修改指针的指向，但可以修改指针所指向区域的值。
* 常量指针：不可以修改指针所指向区域的值，但可以修改指针的指向。
* 常量指针常量：两个都不可以修改
* 小技巧：
* 哪个在前面，哪个就不可以修改

### 引用和函数的关系
* 引用的本身是指针常量
* 值传递：传递的对象的副本
* 引用传递：传递的是对象本身
* 值返回：转递的是函数内对象的副本
* 引用返回：转递的是函数内对象的本身

### 拷贝构造为什么要传入const修饰对象的引用
* 假如我们的拷贝构造函数是这样的A(A a)；，这是一个值传递，那我们会调用一个拷贝构造函数，而调用的拷贝构造函数还是值传递，那么我们就需要继续调用拷贝构造函数，这样的的话就是子子孙孙无穷匮也，内存就爆炸啦，因此必须传引用。
### 纯虚函数
* 语法
```
virtual void func (int a ) = 0;
```
* 当类中含有纯虚函数，该类也称为抽象类
* 抽象类无法实例化对象
* 何为实例化对象**？？？**
> 无法创建对象
* 子类必须重写纯虚函数，否则也为抽象类
* 虚函数的分文件编写


### 虚析构和纯虚析构
* 用于释放多态使用时，子类的堆区空间   
* 语法

```
虚析构
virtual ~类名(){} 

纯虚析构
virutal ~类名() = 0;
类名 :: ~类名(){}
```
* 为什么需要虚析构
* 通过父类指针释放，会导致只调用父类的析构函数，无法调用子类的析构函数，会导致子类对象可能清理的不干净
* 而虚函数是地址晚绑定的，可以通过多态来调用子类的析构函数

* 虚析构就是也用来解决通过父类指针释放子类对象
* 如果子类中没有堆区数据，可以不写虚析构
* 拥有纯虚析构的类为抽象类，不可实例化对象


### 友元
* 全局函数做友元
* 
### 模板
* temple<>func();
* 多态，优先调用
* 类模板
* 显示调用
* 类模板只有显示类型推导
* 类模板没有自动类型推导
```
stack<int> nums;
```
* 类模板中参数列表可以有默认参数

```
template<class T,class S = int >
class MyStruct
{
      
}
```

### STL
#### 字符串
* 输入含有空格的字符串 