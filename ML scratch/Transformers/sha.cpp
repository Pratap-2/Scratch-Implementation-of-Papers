
#include <bits/stdc++.h>
using namespace std;

// Softmax: normalize each row independently
vector<vector<float>> softmax(vector<vector<float>> scores) {
    for (int i = 0; i < scores.size(); i++) {
        if (scores[i].empty()) continue;

        float maxi = *max_element(scores[i].begin(), scores[i].end());

        float sumi = 0.0f;

        for (int j = 0; j < scores[i].size(); j++) {
            scores[i][j] = exp(scores[i][j] - maxi);
            sumi += scores[i][j];
        }

        for (int j = 0; j < scores[i].size(); j++) {
            scores[i][j] /= sumi;
        }
    }

    return scores;
}


// Scaled dot-product attention
// Q: (n, dk), K: (n, dk), V: (n, dv)
// Output: (n, dv)
// Attention weights: (n, n)

pair<vector<vector<float>>, vector<vector<float>>>
selfattention(
    const vector<vector<float>>& q,
    const vector<vector<float>>& k,
    const vector<vector<float>>& v
) {
    int n = q.size();

    assert(n > 0);
    assert(k.size() == n && v.size() == n);

    int dk = q[0].size();
    int dv = v[0].size();

    assert(dk > 0);
    assert(k[0].size() == dk);

    for (const auto& row : q) assert(row.size() == dk);
    for (const auto& row : k) assert(row.size() == dk);
    for (const auto& row : v) assert(row.size() == dv);

    // Step 1: Calculate QK^T / sqrt(dk)
    vector<vector<float>> scores(
        n, vector<float>(n, 0.0f)
    );

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float dot_product = 0.0f;

            for (int t = 0; t < dk; t++) {//trickly transpose multiplication of K
                dot_product += q[i][t] * k[j][t];
            }

            scores[i][j] = dot_product / sqrt((float)dk);
        }
    }

    // Step 2: Convert scores into attention probabilities
    vector<vector<float>> attention_weights = softmax(scores);

    // Step 3: Calculate attention_weights * V
    vector<vector<float>> output(
        n, vector<float>(dv, 0.0f)
    );

    for (int i = 0; i < n; i++) {
        for (int t = 0; t < dv; t++) {
            for (int j = 0; j < n; j++) {
                output[i][t] +=
                    attention_weights[i][j] * v[j][t];
            }
        }
    }

    return {output, attention_weights};
}


int main() {
    vector<vector<float>> Q = {
        {1, 0},
        {0, 1},
        {1, 1}
    };

    vector<vector<float>> K = {
        {1, 0},
        {0, 1},
        {1, 1}
    };

    vector<vector<float>> V = {
        {10, 0},
        {0, 10},
        {5, 5}
    };

    auto [output, weights] = selfattention(Q, K, V);

    cout << "Attention weights:\n";
    for (const auto& row : weights) {
        for (float x : row) {
            cout << fixed << setprecision(4) << x << " ";
        }
        cout << "\n";
    }

    cout << "\nAttention output:\n";
    for (const auto& row : output) {
        for (float x : row) {
            cout << fixed << setprecision(4) << x << " ";
        }
        cout << "\n";
    }

    return 0;
}
