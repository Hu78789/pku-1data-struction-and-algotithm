#include <iostream>
#include <string>
#include <queue>
#include <vector>
#include <algorithm>
#include <map>
#include <iterator>
using namespace std;
typedef vector<char> HuffmanCode;
typedef map<char, HuffmanCode> HuffmanTable;
class INode{
    public:
        const int freq;
        virtual ~INode(){}
    protected:
        INode(int f):freq(f){}
};
class InternalNode: public INode
{
    public:    
        INode* left;
        INode* right;
        InternalNode(INode* l, INode* r):
            INode(l->freq + r->freq), left(l), right(r){}
            ~InternalNode(){
                delete left;
                delete right;
            }
            
};
class LeafNode: public INode{
    public:
        const char c;
        LeafNode(int f,char c):INode(f),c(c){}
};
struct NodeCmp{
    bool operator()(const INode* lhs,const INode* rhs) const{return lhs->freq > rhs->freq;}
};
INode* BuildTree(map<char,int> frequs){
    priority_queue<INode*, vector<INode*>, NodeCmp> trees;
    map<char,int>::iterator it = frequs.begin();
    for (; it != frequs.end();it++){
        trees.push(new LeafNode(it->second,it->first));
    }
    while (trees.size() > 1){
        INode* childR = trees.top();
        trees.pop();
        INode* childL = trees.top();
        trees.pop();
        trees.push(new InternalNode(childL,childR));
    }
    return trees.top();
};
void GenerateCode(const INode* node,const HuffmanTable& prefix, HuffmanCode& outCodes){
    
}


