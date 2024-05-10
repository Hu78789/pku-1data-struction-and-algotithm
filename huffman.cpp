#include<bits/stdc++.h>
using namespace std;
struct HuffmanNode{
    char data;
    int freq;
    HuffmanNode *left;
    HuffmanNode *right;
    HuffmanNode(char data,int freq): data(data),freq(freq),left(nullptr),right(nullptr){}
};
struct CmpNodes{
    bool operator()(HuffmanNode*a,HuffmanNode*b){
        return a->freq > b->freq;
    }
};
HuffmanNode* buildHuffmanTree(const unordered_map<char,int>&freqMap){
    priority_queue<HuffmanNode*,vector<HuffmanNode*>,CmpNodes> pq;
    for (const auto& pair : freqMap){
        pq.push(new HuffmanNode(pair.first,pair.second));
        }
    while (pq.size()>1){
        HuffmanNode* left = pq.top();pq.pop();
        HuffmanNode* right = pq.top();pq.pop();
        HuffmanNode* newNode = new HuffmanNode('$',left->freq+right->freq);
        newNode->left = left;
        newNode->right = right;
        pq.push(newNode);
    }
    return pq.top();    
}
void generateHuffmanCode(HuffmanNode* root,string code,unordered_map<char,string>& huffmanCodes){
    if (root == nullptr) return;
    if (!root->left && !root->right){
        huffmanCodes[root->data] = code;
    }
    generateHuffmanCode(root->left,code+"0",huffmanCodes);
    generateHuffmanCode(root->right,code + "1",huffmanCodes);
}
unordered_map<char,string> huffmanEncoding(const unordered_map<char,int>&freqMap){
    HuffmanNode* root = buildHuffmanTree(freqMap);
    unordered_map<char,string> huffmanCodes;
    generateHuffmanCode(root,"",huffmanCodes);
    return huffmanCodes;
}
string decodeHuffmanData(HuffmanNode *root,const string& str){
    string ans = "";
    HuffmanNode*curr = root;
    for (char s:str){
        if (s == '0'){
            curr = curr->left;    
        }else curr = curr->right;
    }
    if(curr->left == nullptr && curr->right == nullptr){
        ans += curr->data;
        curr = root;
    }
    return ans;
}


int main() {
    // 测试字符频率
    unordered_map<char, int> freqMap = {
        {'a', 5}, {'b', 9}, {'c', 12}, {'d', 13},
        {'e', 16}, {'f', 45}
    };

    // 生成哈夫曼编码
    unordered_map<char, string> huffmanCodes = huffmanEncoding(freqMap);

    // 输出结果
    cout<<"Huffman Codes:\n";
    for(const auto& pair : huffmanCodes) {
        cout << pair.first << " : " << pair.second << endl;
    }
    string a="0";
    string result = decodeHuffmanData(buildHuffmanTree(freqMap),a);
    cout<<result<<endl;

    return 0;
}
