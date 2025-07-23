#include<bits/stdc++.h>
using namespace std;
using i64 = long long;
using u64 = unsigned long long;
using u32 = unsigned;
 
using u128 = unsigned __int128;
using i128 = __int128;
#define ll long long
#define int long long 
// vector<int> out;
const ll mx=1e18;

ll gcd (ll a, ll b) {
    if (b == 0)
        return a;
    else
        return gcd (b, a % b);
}
const int MOD = 998244353;

int modPower(int base, int exp, int mod) {
    int result = 1;
    long long b = base;
    while (exp > 0) {
        if (exp & 1) result = (long long)result * b % mod;
        b = (b * b) % mod;
        exp >>= 1;
    }
    return result;
}

int modInverse(int y, int mod = MOD) {
    return modPower(y, mod - 2, mod);
}

int modularDivide(int x, int y, int mod = MOD) {
    return (int)((long long)x * modInverse(y, mod) % mod);
}
int32_t main() {
    int t;
    cin >> t;
    while (t--) {
        int n, k;
        cin >> n >> k;
        vector<int> v(n);
        for (int i = 0; i < n; i++) cin >> v[i];
        if (k >= 3) {
            cout << 0 << endl;
            continue;
        }
        sort(v.begin(), v.end());
        int d = v[0];
        for (int i = 0; i < n - 1; i++) d = min(d, v[i + 1] - v[i]);
        if (k == 1) {
            cout << d << endl;
            continue;
        }
        for (int i = 0; i < n; i++) for (int j = 0; j < i; j++) {
            int tt = v[i] - v[j];
            int p = lower_bound(v.begin(), v.end(), tt) - v.begin();
            if (p < n) d = min(d, v[p] - tt);
            if (p > 0) d = min(d, tt - v[p - 1]);
        }
        cout << d << endl;
    }
}