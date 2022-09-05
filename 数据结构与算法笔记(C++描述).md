# 数据结构与算法笔记(C++描述)

## 时间复杂度

## 数据结构

### 链表

#### 链表的结构

```C++
struct _node
{
	int data;							//数据域
	struct _node* next;					//指针域
};
typedef struct _node node;
```

#### 链表的创建

```C++
//创建一个节点数为 N 的链表, 返回头节点的地址
node* linkNodeCreat(int N)//尾插法
{
	node* head = NULL;
	node* p = (node*)malloc(sizeof(node));
	p->data = N;
	p->next = NULL;
	head = p;
	
	for (int i = 0; i < N; i++)
	{
		node* p = (node*)malloc(sizeof(node));
		scanf("%d", &p->data);
		p->next = NULL;

        //找到链表的尾巴
		node* last = head;
		while (last->next != NULL)
            last = last->next;

		last->next = p;
	}
	return head->next;
}
```

#### 链表的打印

```C++
//链表的打印
void linkNodePrint(node* head)
{
	if (head == NULL) 
        printf("该链表没有节点");
	for (node* p = head; p != NULL ; p = p->next)
        printf("%d ", p->data);
	printf("\n");
}
```

#### 链表的查找

```C++
//链表的查找, 返回查找元素的索引值, 没有找到返回-1
int linkNodeFind(node* head,int x)
{
	int index = -1;
	for (node* p = head; p != NULL; p = p->next)
	{
		index++;
		if (p->data == x)
		{
			index++;
			//因为索引是从-1开始计数, 所以最后输出的时候要+1
			break;
		}
	}
	return index;
}
```

#### 链表的添加

```C++
//链表的添加, x为添加节点的数据, index表示添加节点的索引, 返回头节点的地址
node* linkNodeAdd(node* head, int x, int index)
{
	node* temp = (node*)malloc(sizeof(node));
	temp->next = head;
	head = temp;

	//自己人为地添加头节点, 其中添加的头节点的数值域不包含任何数据
	//这样做可以减少判断头指针是否发生改动或者头指针为空的操作
	//最后返回的时候, 只要返回头节点的下一个节点就行了

	node* p = head;
	for (int i = 1; i < index; p = p->next,i++)
	{
		if (p == NULL)
		{
			printf("索引超出当前链表的最大长度\n");
			break;
		}
	}
	node* q = (node*)malloc(sizeof(node));
	q->data = x;
	q->next = p->next;

	p->next = q;
	return head->next;
}
```

#### 链表的删除

```C++
//链表的删除, x为要删除节点的数据, 返回头节点的地址
node* linkNodeRemove(node* head, int x)
{
	node* p = head;
	head = (node*)malloc(sizeof(node));
	head->next = p;
	//添加头节点
	//理由和linkNodeAdd()一样, 不在解释

	//双指针法
	//p所指向的是原来链表的头节点
	//现在的链表的第二个节点
	for (node* q = head; p !=NULL; p = p->next,q = q->next)
	{
		if (p->data == x)
		{
			q->next = p->next;
			free(p);
			return head->next;//返回头节点的下一个节点
		}
	}
	printf("没有找到指定元素\n");
	return head->next;

}
```

---

### 栈

#### 数组实现



#### 链表实现

---

### 队列

#### 数组实现



#### 链表实现

---

### 树

#### 前言

#### 树的结构

```C++
struct _node
{
	int data;
	struct _node* left;
	struct _node* right;
};
```

#### 二叉树的构建

```C++
node* binaryTreeCreatNode(node* head ,int val)
{
	if (head == NULL)
	{
		node* p = (node*)malloc(sizeof(node));
		p->data = val;
		p->left = p->right = NULL;
		return p;
	}
	else
	{
		if (head->data > val) head->left = binaryTreeCreatNode(head->left, val);
		if (head->data < val) head->right = binaryTreeCreatNode(head->right, val);
	}
	
	return head;
}

node* binaryTreeCreat()
{
	int N;
	scanf("%d", &N);
	node* head = NULL;
	for (int i = 0; i < N; i++)
	{
		int x;
		scanf("%d", &x);
		head = binaryTreeCreatNode(head,x);
	}
	return head;
}
```

#### 二叉树的删除

```C++
node* binaryTreeFindMin(node* head)
{
	node* p = head;
	while (p->left != NULL)
	{
		p = p->left;
	}
	return p;
}

node* binaryTreeDelete(node* head, int x)
{
	if (head == NULL) cout << "删除元素找不到" << endl;
	else if (head->data > x)
        head->left = binaryTreeDelete(head->left, x);
	else if(head->data < x)
        head->right = binaryTreeDelete(head->right, x);
	else
	{
		if (head->left != NULL && head->right != NULL)
		{
			node*temp = binaryTreeFindMin(head->right);
			head->data = temp->data;
			head->right = binaryTreeDelete(head->right, head->data);
		}
		else
		{
			node* p = head;
			if (head->left != NULL) head = head->left;
			else if (head->right != NULL)head = head->right;
			else head = NULL;
			free(p);
		}
	}
	return head;
}


```



#### 树的遍历

##### 前序遍历

```C++
void binaryTreePreorderTraversal(node* head)
{
	if (head == NULL) return;
	else
	{
        printf("%d ", head->data);//这一行在前面就是前序遍历
		binaryTreePrint(head->left);
		binaryTreePrint(head->right);
	}
}
```

##### 中序遍历

```C++
void binaryTreeInorderTraversal(node* head)
{
	if (head == NULL) return;
	else
	{
		binaryTreePrint(head->left);
        printf("%d ", head->data);//这一行在中间就是中序遍历
		binaryTreePrint(head->right);
	}
}
```

##### 后序遍历

```C++
void binaryTreePostorderTraversal(node* head)
{
	if (head == NULL) return;
	else
	{
		binaryTreePrint(head->left);
		binaryTreePrint(head->right);
        printf("%d ", head->data);//这一行在后面就是后序遍历
	}
}
```

##### 层序遍历

```C++
void binaryTreeSequenceTraversal(node* head)
{
	queue<node*> que;//创建队列
	if (head != NULL) que.push(head);
	while (!que.empty())
	{
		node* p = que.front();
		cout << p->data << ' ';
		que.pop();
		if (p->left != NULL) que.push(p->left);
		if (p->right != NULL) que.push(p->right);
	}
	cout << endl;
	return;
}
```

#### 哈夫曼编码

#### 平衡二叉树

<u>**暂无**</u>

---

### 图

#### 前言

```C++
const int SIZE = 10;//表示邻接矩阵的大小
const int INF = 99999;//表示机器的最大值
```

#### 图的结构

##### 邻接矩阵

```C++
struct Graph
{
	int edge[SIZE][SIZE];
	int point[SIZE][SIZE];
	int numberPoint, numberEdge;
};
```



#### 图的创建

##### 邻接矩阵

```C++
void graphCreate(Graph* G)//创建无向有权图
{
    //输入提示信息
	cout << "please input the number of points" << endl;
	cout << "please input the number of edges" << endl;
	cin >> G->numberPoint >> G->numberEdge;
	for (int i = 0; i < G->numberPoint; i++)
	{
		for (int j = 0; j < G->numberPoint; j++)
		{
			G->edge[i][j] = -INF;
            //将无边用负无穷表示
            //在之后的判断有无边中，可以减少代码量
			if (i == j) G->edge[i][j] = 0;
		}
	}

	for (int i = 0; i < G->numberEdge; i++)
	{
		int vi, vj, weight;
		cout << "please input the information of this edge" << endl;
		cout << "eg: 0 3 1" << endl;
		cin >> vi >> vj >> weight;
		
		G->edge[vi][vj] = weight;
		G->edge[vj][vi] = weight;
	}
	system("cls");//清屏
}
```



#### 图的打印

```C++
void graphPrint(Graph* G)
{
	cout << endl << endl;
	for (int i = 0; i < G->numberPoint; i++)
		cout << '\t' << i;
	for (int i = 0; i < G->numberPoint; i++)
	{
		cout << endl << i;
		for (int j = 0; j < G->numberPoint; j++)
			cout << '\t' << G->edge[i][j];
	}
    cout<<endl<<endl;
}
```



#### 图的遍历

##### DFS算法(深度优先算法)

###### 递归算法

```C++
//全局变量
bool dist[MATRIX_SIZE]; //用于记录节点是否访问过
//由于dist是全局数组，未初始化时，默认值为false

void graphDFS(Graph*G, int vertex)//从顶点vertex开始遍历
{
	dist[vertex] = true;//标记该节点已访问过
	cout << "正在访问节点" << vertex << endl;

	for (int i = 0; i < G->numberPoint; i++)
		if (dist[i] == false && G->edge[vertex][i] > 0)
            //判断下一个节点是否未访问过
       		//判断下一个节点与该节点是否有边链接
			graphDFS(G, i);
}
```
###### 非递归算法

* 代码和BFS算法类似
* 只不过是把队列换成了栈

```C++
void graphDFS(Graph*G, int vertex)//从顶点vertex开始遍历
{
	stack<int> st;	//定义栈
	bool visit[SIZE] = { false };
	st.push(vertex);
	visit[vertex] = true;//标记该节点已访问过
	
	while (!st.empty())
	{
		int k = st.top();
		cout << "正在访问节点" << k << endl;
		st.pop();
		for (int i = 0; i < G->numberPoint; i++)
		{
			if (visit[i] == false && G->edge[k][i] > 0)
			{
				visit[i] = true;//标记该节点已访问过
				st.push(i);
			}
		}
	}
}
```



##### BFS算法(广度优先算法)

```C++
void graphBFS(Graph* G,int vertex)//从顶点vertex开始遍历
{
    //初始化
	queue<int> que;	//定义队列
	bool visit[SIZE] = { false };
	que.push(vertex);
	visit[vertex] = true;

	while (!que.empty())
	{
		int k = que.front();
		cout << "正在访问节点" << k << endl;
		que.pop();
		for (int i = 0; i < G->numberPoint; i++)
		{
			if (G->edge[k][i] > 0 && visit[i] == false)
			{
				visit[i] = true;
				que.push(i);
			}
		}
	}
}
```



#### 最小生成树

##### Prim算法

###### 前言

* **注意**
  * 

###### 核心代码

```C++
void graphPrim(Graph* G)
{
	int parent[SIZE];
	int lowcost[SIZE];
	for (int i = 0; i < G->numberPoint; i++)
    {
        parent[i] = i;
		if (G->edge[0][i] >= 0)
			lowcost[i] = G->edge[0][i];
		else
			lowcost[i] = -G->edge[0][i];
    }

    //注意此处循环变量的初始值是1
	for (int i = 1; i < G->numberPoint; i++)
	{
		int minVertex = -1;
		int minWeight = INF;
		for (int j = 0; j < G->numberPoint; j++)
		{
			if (lowcost[j] > 0 && minWeight > lowcost[j])
			{
				minVertex = j;
				minWeight = lowcost[j];
			}
		}
		cout << parent[minVertex] << " <=> " << minVertex << '\t' << minWeight << endl;
		lowcost[minVertex] = 0;//标记此顶点

		for (int j = 0; j < G->numberPoint; j++)
		{
			if (G->edge[minVertex][j] > 0)
            {
                if(lowcost[j] > G->edge[minVertex][j])
                {
                    lowcost[j] = G->edge[minVertex][j];
                    parent[j] = minVertex;
                }
            }
		}
	}
}
```



##### Kruskal算法

###### 前言

* 步骤
  * 编写一个测试用例
  * 定义边的结构体, 包含的开头和结尾, 还有边的权重
  * 编写并查集的查找函数( disjointSetFind )
  * 初始化并查集数组
  * 初始化边集数组
  * 对边集数组进行排序
  * 将边集数组打印一下
* 并查集的作用
    * 用于判断该图是否成环
  
* 注意事项
  * 在初始化边集数组的时候
    * 两个循环变量的初始值不相同

###### 边的结构

```c++
struct Edge
{
    int begin;
	int weight;
	int end;
};
```



###### 并查集的查找

```C++
int disjointSetFind(int parent[], int x)
{
	return parent[x] == x ? x : disjointSetFind(parent, parent[x]);
}
```

###### 核心算法

```C++
void graphKruskal(Graph* G)
{
    //初始化
	int parent[SIZE];
	vector<Edge> vec;
	for (int i = 0; i < G->numberPoint; i++)
	{
		for (int j = i; j < G->numberPoint; j++)
		{
			if (G->edge[i][j] > 0)
			{
				Edge temp;
				temp.begin = i;
				temp.end = j;
				temp.weight = G->edge[i][j];
				vec.push_back(temp);
			}
		}
	}
    
    //将边根据权值大小按照从小到大的顺序排序
	sort(vec.begin(), vec.end(), cmp);
    
	for (auto e : vec)
	{
		int n = disjointSetFind(parent, e.begin);
		int m = disjointSetFind(parent, e.end);
		if (n != m)
		{
			parent[m] = n;
			cout << e.begin << " <=> " << e.end 
                << '\t' << e.weight << endl;
		}
	}
}

```



#### 最短路径问题

##### Dijkstra算法

```c++
//从图G的vertex顶点到其余顶点的最短路径
void graphDijkstra(Graph* G, int vertex)
{
	int parent[SIZE];
	int lowpath[SIZE];
	bool visit[SIZE];
	for (int i = 0; i < G->numberPoint; i++)
	{
		
		parent[i] = vertex;
		lowpath[i] = G->edge[vertex][i];
		if (lowpath[i] == -INF)
			lowpath[i] *= -1;
		visit[i] = false;
	}
	visit[vertex] = true;
    
    //注意此时i的初始值是1 
	for (int i = 1; i < G->numberPoint; i++)
	{
		int minVertex = -1;
		int minWeight = INF;
		for (int j = 0; j < G->numberPoint; j++)
		{
			if (visit[j] == false && minWeight > lowpath[j])
			{
				minWeight = lowpath[j];
				minVertex = j;
			}
		}
		visit[minVertex] = true;

		for (int j = 0; j < G->numberPoint; j++)
		{
			if (visit[j] == false && G->edge[minVertex][j] > 0)
			{
				if (lowpath[j] > minWeight + G->edge[minVertex][j])
				{
					lowpath[j] = minWeight + G->edge[minVertex][j];
					parent[j] = minVertex;
				}
			}
		}
	}
}
```



##### Floyd算法

```C++
void graphFloyd(Graph* G)
{
    //初始化
	int lowpath[SIZE][SIZE];//记录最短路径的权值和
	int parent[SIZE][SIZE];//记录最短路径的路径
	for (int i = 0; i < G->numberPoint; i++)
	{
		for (int j = 0; j < G->numberPoint; j++)
		{
			lowpath[i][j] = G->edge[i][j];
			parent[i][j] = j;
		}
	}
	
	for (int k = 0; k < G->numberPoint; k++)			//中间顶点
	{
		for (int i = 0; i < G->numberPoint; i++)		//开始顶点
		{
			for (int j = 0; j < G->numberPoint; j++)	//结尾顶点
			{
				//经过中间顶点的路径比原来两点之间的路径更短
				if (lowpath[i][j] > lowpath[i][k] + lowpath[k][j])
				{
					lowpath[i][j] = lowpath[i][k] + lowpath[k][j];
					parent[i][j] = parent[i][k];
				}
			}
		}
	}
}
```



#### 有向无环图

##### 拓扑排序==（未修改）==

```C++
void graphTopologySort(Graph* G)
{
	int indegree[MATRIX_SIZE];
	queue<int> que;
	int visit[MATRIX_SIZE];
	for (int i = 0; i < G->numberPoint; i++)
	{
		indegree[i] = 0;
		//visit[i] = false;
		for (int j = 0; j < G->numberPoint; j++)
		{
			if (G->edge[j][i] != 0 && G->edge[j][i] != INF)
				indegree[i] ++;
		}
		if (indegree[i] == 0)que.push(i);
	}

	while (!que.empty())
	{
		int temp = que.front();
		que.pop();
		cout << temp << " => ";
		for (int i = 0; i < G->numberPoint; i++)
		{
			if (G->edge[temp][i] != 0 && G->edge[temp][i] != INF)
			{
				indegree[i] --;
				//注意点,一定要嵌套两重分支语句
				if (indegree[i] == 0)que.push(i);
			}
		}
	}
}
```

##### 关键路径

```C++
void graphCriticalPath(Graph* G)
{

}
```



## 算法

### 查找

#### 二分查找

```C++
//nums是数组，size是数组的大小，target是需要查找的值
int search(int nums[], int size, int target) 
{
    int left = 0;
    int right = size - 1;	
    // 定义了target在左闭右闭的区间内，[left, right]
    while (left <= right) 
    {	
        //当left == right时，区间[left, right]仍然有效
        int middle = left + ((right - left) / 2);
        //等同于 (left + right) / 2，防止溢出
        if (nums[middle] > target) right = middle - 1;	
        //target在左区间，所以[left, middle - 1]
        else if (nums[middle] < target) left = middle + 1;	
        //target在右区间，所以[middle + 1, right]
        else return middle;
        //既不在左边，也不在右边，那就是找到答案了     
    }
    return -1;	//没有找到目标值
}

```

#### 斐波那契查找

```c++
/*构造一个斐波那契数组*/
void Fibonacci(int* F)
{
    F[0] = 0;
    F[1] = 1;
    for (int i = 2; i < MAX_SIZE; ++i)
        F[i] = F[i - 1] + F[i - 2];
}

/*定义斐波那契查找法*/
//a为要查找的数组,n为要查找的数组长度,key为要查找的关键字
int FibonacciSearch(int* a, int n, int key)  
{
    int low = 0;
    int high = n - 1;
    int k = 0;
    int F[MAX_SIZE] = {0};

    Fibonacci(F);						//构造一个斐波那契数组F 

    while (n > F[k] - 1)				//计算n位于斐波那契数列的位置
        ++k;
    int* temp;							//将数组a扩展到F[k]-1的长度
    temp = malloc(sizeof(int)*(F[k]-1));
    memcpy(temp, a, n * sizeof(int));	//将a中的元素进行拷贝至temp

    for (int i = n; i < F[k] - 1; ++i)	//填充数组
        temp[i] = a[n - 1];

    while (low <= high)					//终止条件和二分查找一致
    {
        int mid = low + F[k - 1] - 1;	//递推关系式，获取mid值
        if (key < temp[mid])
        {
            high = mid - 1;
            k -= 1;
        }
        else if (key > temp[mid])
        {
            low = mid + 1;
            k -= 2;
        }
        else
        {
            free(temp);//记得free
            if (mid < n)
                return mid; //若相等则说明mid即为查找到的位置
            else
                return n - 1; //若mid>=n则说明是扩展的数值,返回n-1
        }
    }
    free(temp);//记得free
    return -1;
}
```

###### 注意

适用于数据分布不均匀的情况

#### 哈希表



### 排序

默认为升序

#### 冒泡排序

```c++
void bubbleSort(int arr[], int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < i; j++)
		{
			if (arr[i] < arr[j])
				swap(arr[i], arr[j]);
		}
	}
}
```



#### 插入排序

```C++
void InsectSort(int a[], int n)
{
	for (int i = 1; i < n; i++)
	{
		int temp = a[i];
		int j;
		for (j = i; j >= 1 && a[j -1] > temp;j-=1 )
		{
			a[j] = a[j -1];
			
		}
		a[j] = temp;
	}
}
```



#### 选择排序

```C++
void SelectionSort(int a[], int n)
{
	for (int i = 0; i < n; i++)
	{
		int minindex = i;
		for (int j = i; j < n; j++)
		{
			if (a[minindex] > a[j])
			{
				minindex = j;
			}
		}
		swap(a[i], a[minindex]);

	}

}
```



#### 希尔排序

```C++
void ShellSort(int a[], int n)
{
	int D[6] = { 1,3,7,15,31,65 };
	int num;
	for (num = 0; D[num] < n; num++);
	num--;
	for (; num >= 0; num--)
	{
		for (int i = D[num]; i < n; i++)
		{
			int temp = a[i];
			int j;
			for (j = i; j > D[num] && a[j - D[num]] > temp; )
			{
				a[j] = a[j - D[num]];
				j-= D[num];
			}
			a[j] = temp;
		}
	}
}
```



#### 快速排序

```C++
void QuickSort(int a[], int begin,int end)
{
	if (begin >= end)
	{
		return;
	}
	int temp = a[begin];
	int i = begin;
	int j = end;
	while (i < j)
	{
		while (a[j] >= temp && j > i)
		{
			j--;
		}
		while (a[i] <= temp && j > i)
		{
			i++;
		}
		if (i < j)
		{
			swap(a[i], a[j]);
		}
	}
	swap(a[begin], a[i]);
	QuickSort(a, begin, i - 1);
	QuickSort(a, i+1, end);

}
```



#### 堆排序

```C++
void ShiftHeap(int a[], int point,int n)
{
	int parent, child;
	for (int parent = point; parent * 2 + 1 < n; parent = child)
	{
		child = parent * 2 + 1;
		if ((child + 1 < n) && (a[child] < a[child + 1]))
		{
			child++;
		}
		if (a[parent] < a[child])
		{
			swap(a[parent], a[child]);
		}
		else
		{
			break;
		}
		
	}
}

void HeapSort(int a[], int n)
{
	for (int i = n / 2 - 1; i >= 0; i--)
	{
		ShiftHeap(a, i, n);
	}
	
	for (int i = n - 1 ; i > 0; i--)
	{
		swap(a[0], a[i]);
		ShiftHeap(a, 0, i);
	}
}
```

#### 归并排序



### 高级搜索树

#### B树

#### B+树