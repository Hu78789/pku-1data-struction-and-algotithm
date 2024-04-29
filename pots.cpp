#include<iostream>
#include<cstdio>
#include<queue>
#include<stack>
#include<cstring>
#include<cstdlib>
using namespace std;
int a,b,c;
bool vis[110][110];
/*
Starting From (0,0),End With (C,X) Or (X,C)
Operations:
FILL(1):(A,XB)
FILL(2):(XA,B)
DROP(1):(0,XB)
DROP(2):(XA,0)
POUR(2,1):
POUR(1,2):
*/
struct s{
    int pot1;//F:fill,pour,drop
    int pot2;
    int steps;
    int op;
    int father;
    s(){}
    s(int p1,int p2,int s,int o,int f):pot1(p1),pot2(p2),steps(s),op(o),father(f) {}
};

void Output(int op){
    switch(op){
        case 0:
            cout<<"FILL(1)"<<endl;
            break;
        case 1:
            cout<<"FILL(2)"<<endl;
            break;
        case 2:
            cout<<"DROP(1)"<<endl;
            break;
        case 3:
            cout<<"DROP(2)"<<endl;
            break;
        case 4:
            cout<<"POUR(1,2)"<<endl;
            break;
        case 5:
            cout<<"POUR(2,1)"<<endl;
            break;    
    }
}

int main(){
    ios::sync_with_stdio(false);
    while(cin>>a>>b>>c){
        memset(vis,0,sizeof(vis));//vis中数组值为0
        s q[50000];//s为结构体
        int head,tail;
        head=tail=0;
        vis[0][0]=1;
        q[tail++] = s(0,0,0,-1,-1);
        bool flag=false;
        while (head!=tail){
            s t=q[head];
            if(t.pot1==c||t.pot2==c){
                flag = true;
                break;
            }
            if(!vis[a][t.pot2]){
                vis[a][t.pot2]=1;
                q[tail++] = s(a,t.pot2,t.steps+1,0,head);
            }
            if(!vis[t.pot1][b]){
                vis[t.pot1][b]=1;
                q[tail++] = s(t.pot1,b,t.steps+1,1,head);
            }
            if(!vis[0][t.pot2]){
				vis[0][t.pot2]=1;
				q[tail++]=s(0,t.pot2,t.steps+1,2,head);
				
			}
			if(!vis[t.pot1][0]){
				vis[t.pot1][0]=1;
				q[tail++]=s(t.pot1,0,t.steps+1,3,head);				
			}
            int sum = t.pot1+t.pot2;
            if (sum>b){
                if (!vis[sum-b][b]){
                    vis[sum-b][b]=1;
                    q[tail++]=s(sum-b,b,t.steps+1,4,head);
                }
            }else{
                if(!vis[0][sum]){
                    vis[0][sum]=1;
                    q[tail++]=s(0,sum,t.steps+1,4,head);
                }
            }
            if (sum>a){
                if (!vis[a][sum-a]){
                    vis[a][sum-a] = 1;
                    q[tail++]=s(a,sum-a,t.steps+1,5,head);
                }
            }else{
                if(!vis[sum][0]){
                    vis[sum][0] = 1;
                    q[tail++]=s(sum,0,t.steps+1,5,head);
                }
            }
            head++;

        }
        if(!flag){
            cout<<"impossible"<<endl;
            continue;
        }

        vector<s> res;
        s path=q[head];
        while(path.father!=-1){
            res.push_back(path);
            path = q[path.father];
        }
        cout<<res.size()<<endl;
        for(int i=res.size()-1;i>=0;i--){
            Output(res[i].op);
        }
    }
    return 0;
}