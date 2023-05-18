import numpy as np
import pprint

def attention(Q, K, V):
    S = np.matmul(Q, np.transpose(K))
    rowsum = np.sum(np.exp(S), 1)
    P = np.exp(S) / rowsum.reshape(-1,1)
    O = np.matmul(P, V)
    return O


def flashAttention(Q, K, V, M=None):
    n, d = Q.shape
    if M is None:
        M = n
    bc, br = np.ceil(M/4/d), min(np.ceil(M/4/d), d)
    O, l = np.zeros((n, d)), np.zeros(n)
    tr, tc = np.ceil(n/br), np.ceil(n/bc)
    br, bc, tr, tc = int(br), int(bc), int(tr), int(tc)
    print(M, d, n, br, bc, tr, tc)
    for j in np.arange(tc, dtype=int):
        endj = (j+1)*bc
        Kj, Vj = K[j*bc: endj], V[j*bc: endj]
        for i in np.arange(tr, dtype=int):

    # for i in np.arange(tr, dtype=int):
    #     for j in np.arange(tc, dtype=int):
    #         endj = (j+1)*bc
    #         Kj, Vj = K[j*bc: endj], V[j*bc: endj]

            endi = (i+1)*br
            Qi, Oi, li = Q[i*br: endi], O[i*br: endi], l[i*br: endi]
            Sij = np.matmul(Qi, np.transpose(Kj))
            Pij = np.exp(Sij)
            lij = np.sum(Pij, 1)

            linew = li + lij

            # print(mi.shape, minew.shape, Oi.shape, np.exp(mi-minew).shape)
            Oinew = np.matmul(np.diag(1./linew), np.matmul(np.diag(li), Oi)+np.matmul(Pij,Vj))

            O[i*br: endi] = Oinew
            l[i*br: endi] = linew
        # print(O)
    return O

def main():
    
    # Q, K, V = np.random.randint(0, 100, size=(3,4,3))
    Q, K, V = np.random.randn(3,8,3)
    print(Q.shape, K.shape, V.shape)

    O1 = attention(Q,K,V)
    print(O1.shape)

    print(O1)

    O2 = flashAttention(Q,K,V)
    print(O2.shape)

    print(O2)

    print(((O1-O2)**2).sum())

if __name__=="__main__":
    main()

    