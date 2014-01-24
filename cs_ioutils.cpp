#include <stdio.h>
#include <math.h>
#include <iostream.h>
#include "cs.h"
#include "cs_ioutils.h"

//
// cs_LoadFromFile
// Loads matrix from file
//
int cs_LoadBinaryNetworkFromFile(FILE *cFile, cs **vA, unsigned long int M, unsigned long int N, int nnz, int HeaderLines, bool ToSymm) {
	char lString[1024];
	unsigned long int lI, lJ,nnzmax;

	// Skip header lines (if any)
	for (int i = 0; i < HeaderLines; i++) {
		if (fscanf(cFile, "%s", lString) == 0) {
			fclose(cFile);
			return 1;
		}
	}

	// If M==0 or N==0, assume the number of vertices is in the first non-header line of the file
	if( (M==0) || (N==0) )
	{
		if ( fscanf(cFile,"%li %li",&M,&nnzmax)<2 ) {
			fclose(cFile);
			return 1;
		}
		N = M;
		if( ToSymm )
			nnzmax = 2*nnzmax;
	}

	// Allocate memory for matrix
	*vA = cs_spalloc(M, N, nnzmax, 1, 1);

	// Read in matrix entries
	if (!ToSymm) {
		while (fscanf(cFile, "%li %li", &lI,&lJ) == 2) {
			cs_entry(*vA, lI-1, lJ-1, 1.0);
		}
	} else {
		while (fscanf(cFile, "%li %li", &lI, &lJ) == 2) {
			cs_entry(*vA, lI, lJ, 1.0);
			if (lI != lJ) cs_entry(*vA, lJ, lI, 1.0);
		}
	}

	// Compress the matrix
	*vA = cs_compress(*vA);

	//cout << "\n vA before duplicate averaging:\n";
	//cs_print(*vA,0);

	if (ToSymm) {
		// Average duplicate entries
		cs_ave_dupl(*vA);
	}

	//cout << "\n vA after duplicate averaging:\n";
	//cs_print(*vA,0);

	return 0;
}

int cs_LoadFromFile(const char *cFileName, cs **vA, unsigned long int M, unsigned long int N, int nnz, int HeaderLines, bool ToSymm) {
	FILE *lFile = NULL;

	// Open file
	if ((lFile = fopen(cFileName, "rt")) == NULL) return 1;

	cs_LoadFromFile(lFile, vA, M, N, nnz, HeaderLines, ToSymm);

	fclose(lFile);

	return 0;
}

int cs_LoadFromFile(FILE *cFile, cs **vA, unsigned long int M, unsigned long int N, int nnz, int HeaderLines, bool ToSymm) {
	char lString[1024];
	unsigned long int lI, lJ;
	double lS;

	// Skip header lines (if any)
	for (int i = 0; i < HeaderLines; i++) {
		if (fscanf(cFile, "%s", lString) == 0) {
			fclose(cFile);
			return 1;
		}
	}

	// If M==0 or N==0, assume the number of vertices is in the first non-header line of the file
	if( (M==0) || (N==0) )
	{
		if ( fscanf(cFile,"%li",&M)==0 ) {
			fclose(cFile);
			return 1;
		}
		N = M;
		fscanf(cFile,"%s",lString);
	}

	// Allocate memory for matrix
	*vA = cs_spalloc(M, N, nnz, 1, 1);
	// Read in matrix entries
	if (!ToSymm) {
		while (fscanf(cFile, "%li %li %lf", &lI, &lJ, &lS) == 3) {
			cs_entry(*vA, lI, lJ, lS);
		}
	} else {
		while (fscanf(cFile, "%li %li %lf", &lI, &lJ, &lS) == 3) {
			cs_entry(*vA, lI, lJ, lS);
			if (lI != lJ) cs_entry(*vA, lJ, lI, lS);
		}
	}
	// Compress the matrix
	*vA = cs_compress(*vA);

	return 0;
}

int sparsedemo_01(void) {
	cs *T, *A, *Eye, *AT, *C, *D, *V, *W;
	int i, m, k;
	double *norms;

	T = cs_load(stdin); /* load triplet matrix T from stdin */
	A = cs_compress(T);
	cs_print(A, 0);

	norms = cs_col2norms(A);

	if (!norms) return 0;
	printf("\n Column norms:\n ");
	for (k = 0; k < A->n; k++)
		printf("%f ", norms[k]);
	printf("\n");

	V = cs_spalloc(T->m, 1, T->m, 1, 1);
	for (k = 0; k < A->m; k++)
		cs_entry(V, k, 0, (double) (k + 1));

	W = cs_compress(V);

	double *ips;
	cs_scolip(A, W, &ips);

	printf("\n Inner products:\n");
	for (k = 0; k < T->n; k++)
		printf("%f ", ips[k]);
	;

	printf("\n");

	printf("\n A column:\n");
	cs *Z;
	cs_getcol(A, 1, &Z);
	cs *ZZ;
	cs_copy(Z, &ZZ);
	cs_print(ZZ, 0);

	cs_spfree(ZZ);
	cs_spfree(Z);
	cs_spfree(V);
	cs_spfree(W);
	cs_spfree(T);
	cs_spfree(A);

	return 0;

	printf("T:\n");
	cs_print(T, 0); /* print T */
	A = cs_compress(T); /* A = compressed-column form of T */
	printf("A:\n");
	cs_print(A, 0); /* print A */
	cs_spfree(T); /* clear T */

	AT = cs_transpose(A, 1); /* AT = A' */
	printf("AT:\n");
	cs_print(AT, 0); /* print AT */

	m = A ? A->m : 0; /* m = # of rows of A */
	T = cs_spalloc(m, m, m, 1, 1); /* create triplet identity matrix */
	for (i = 0; i < m; i++)
		cs_entry(T, i, i, 1);
	Eye = cs_compress(T); /* Eye = speye (m) */
	cs_spfree(T);

	C = cs_multiply(A, AT); /* C = A*A' */
	D = cs_add(C, Eye, 1, cs_norm(C)); /* D = C + Eye*norm (C,1) */
	printf("D:\n");
	cs_print(D, 0); /* print D */

	cs_spfree(A); /* clear A AT C D Eye */
	cs_spfree(AT);
	cs_spfree(C);
	cs_spfree(D);
	cs_spfree(Eye);

	return (0);
}

//
// cs_gsqr
//
// Gram-Schmidt with pivoting
//
int cs_gsqr(const cs *A) {
	int k, l, n, CurMaxColIdx;
	cs *Q, *R, *Qcol = NULL, *Rcol;
	double *ColNorms;
	double *ips;
	double sum;

	if (!CS_CSC (A) || !A->x) return (-1); /* check inputs */
	n = A ? A->n : 0;

	// Compute the 2-norms squared of the columns of A
	ColNorms = cs_col2norms(A);

	for (k = 0; k < n; k++) {
		// Find the column with the largest norm
		cs_vecfindmax(ColNorms, n, &CurMaxColIdx);

		// Extract new canditate column for Q
		if (Qcol) {
			cs_spfree(Qcol);
			Qcol = NULL;
		}

		cs_getcol(A, CurMaxColIdx, &Qcol);

		// Compute projection on the Q's already constructed
		if (Q) cs_scolip(Q, Qcol, &ips);

		// Subtract from the candidate its projection onto current Q
		cs_subcols(A, &Qcol, ips);

		// Get rid of entries below precision

		// Insert column in Q
		Q->p[k + 1] = Qcol->p[1];
		for (l = 0, sum = 0; l < k; l++)
			sum += Q->p[l];
		for (l = 0; l < Qcol->p[1]; l++) {
			//Q->i[l] = Qcol
		}
	}

	return 0;
}

/* Maximum 2-norm squared of columns of sparse matrix = max (sum (abs (A))), largest column sum */
double cs_max2norm(const cs *A) {
	CS_INT p, j, n, *Ap;
	CS_ENTRY *Ax;
	double norm = 0, s, entry2;

	if (!CS_CSC (A) || !A->x) return (-1); /* check inputs */

	n = A->n;
	Ap = A->p;
	Ax = A->x;

	for (j = 0; j < n; j++) {
		for (s = 0, p = Ap[j]; p < Ap[j + 1]; p++) {
			entry2 = CS_ABS (Ax [p]);
			s += entry2 * entry2;
		}
		norm = CS_MAX (norm, s);
	}

	return (norm);
}

/* 2-norms squared of columns of sparse matrix = max (sum (abs (A))), largest column sum */

double *cs_col2norms(const cs *A) {
	CS_INT p, j, n, *Ap;
	CS_ENTRY *Ax;
	double s, *norm, entry2;

	if (!CS_CSC (A) || !A->x) return NULL; /* check inputs */
	n = A->n;
	Ap = A->p;
	Ax = A->x;
	norm = (double*) malloc(n * sizeof(double));
	if (!norm) return NULL;

	for (j = 0; j < n; j++) {
		for (s = 0, p = Ap[j]; p < Ap[j + 1]; p++) {
			entry2 = CS_ABS (Ax [p]);
			s += entry2 * entry2;
		}
		norm[j] = s;
	}

	return (norm);
}

// Sums of columns of sparse matrix
double *cs_colsums(const cs *A) {
	CS_INT p, j, n, *Ap;
	CS_ENTRY *Ax;
	double s, *norm;

	if (!CS_CSC (A) || !A->x) return NULL; /* check inputs */
	n = A->n;
	Ap = A->p;
	Ax = A->x;
	norm = (double*) malloc(n * sizeof(double));
	if (!norm) return NULL;

	for (j = 0; j < n; j++) {
		for (s = 0, p = Ap[j]; p < Ap[j + 1]; p++) {
			s += Ax[p];
		}
		norm[j] = s;
	}

	return (norm);
}

/* Maximum 2-norm squared of columns of sparse matrix = max (sum (abs (A))), largest column sum */
int cs_vecfindmax(const double *v, const int vlen, int *maxidx) {
	if (vlen == 0) return 0;
	int k;
	double max;

	*maxidx = 0;
	max = v[0];

	for (k = 1; k < vlen; k++)
		if (v[k] > max) *maxidx = k;

	return 1;
}

// Add a column to a CS matrix, knowing that it is, together with all the subsequent

// columns, empty

int cs_insertlastcolumn(cs *T, const CS_INT *i, const CS_INT j, const CS_ENTRY *x, const CS_INT n)

{
	CS_INT k;

	// This is the slow version
	for (k = 0; k < n; k++)
		cs_entry(T, i[k], j, x[k]);

	return 1;
}

// Computes inner products between the columns of A and v, where v is a sparse vector

int cs_scolip(const cs*A, const cs*v, double** sum)

{
	int m, l, j, curvidx;

	if (!CS_CSC (A) || !A->x) return 0; /* check inputs */
	if (!CS_CSC (v) || !v->x) return 0; /* check inputs */

	if (A->m != v->m) return 0;
	m = A->m;

	*sum = (double*) malloc(m * sizeof(double));

	// Run through the columns of A

	for (j = 0; j < m; j++) {
		// Run through the nonzero entries in the column of A
		curvidx = 0;

		for (l = A->p[j]; l < A->p[j + 1]; l++) {
			if ((A->i[l]) < (v->i[curvidx])) continue;

			while ((A->i[l]) > (v->i[curvidx])) {
				if (++curvidx >= v->p[1]) break;
			}

			if ((A->i[l]) == (v->i[curvidx])) (*sum)[j] += A->x[l] * v->x[curvidx];

			curvidx++;
		}
	}

	return 1;
}

// Extract a column from a matrix
int cs_getcol(const cs*A, int colidx, cs **col) {
	cs *newcol;

	if (!CS_CSC (A) || !A->x) return 0; /* check inputs */

	if (colidx > A->n) return 0;

	int k, l, nentries;

	nentries = A->p[colidx + 1] - A->p[colidx];
	newcol = cs_spalloc(A->m, 1, nentries, 1, 0);
	newcol->p[0] = 0;
	newcol->p[1] = nentries;

	for (k = A->p[colidx], l = 0; k < A->p[colidx + 1]; k++, l++) {
		newcol->i[l] = A->i[k];
		newcol->x[l] = A->x[k];
	}

	*col = newcol;

	return 1;
}

// Subtracts from the column C the columns of A, each multiplied by b
int cs_subcols(const cs*A, cs**Col, double*b) {
	cs *lCurACol, *lColNew;
	int k;

	// Loop through the columns of A
	for (k = 0; k < A->n; k++) {
		// Get the column of A
		cs_getcol(A, k, &lCurACol);

		// Subtract it from v, after rescaling by b[k]
		lColNew = cs_add(lCurACol, *Col, -b[k], 1);

		// Clean memory and update v
		cs_free(*Col);
		*Col = lColNew;
	}

	return 1;
}

// Copy a matrix
int cs_copy(const cs*A, cs **B) {
	cs *zeroMatrix = cs_spalloc(A->m, A->n, A->nzmax, 1, (A->nz > 0 ? 1 : 0));
	*B = cs_add(A, zeroMatrix, 1, 0);
	cs_spfree(zeroMatrix);

	return 1;
}

// Apply exponential function to entries of a matrix
int cs_applyexp(cs *A, double sigma) {
	for (int k = 0; k < A->nzmax; k++)
		A->x[k] = exp(-((A->x[k]) / sigma));

	return 0;
}

// Maps a symmetric matrix A to (1/sqrt(D))*A*(1/sqrt(D))
int cs_symminvsqrtdeg(cs *A) {
	if (!CS_CSC (A) || !A->x) return (-1); /* check inputs */

	// First of all find the degree, by computing the sum of each column
	double *colsums = cs_colsums(A);
	// Compute the inverse square root
	for (int k = 0; k < A->m; k++)
		if (fabs(colsums[k]) > 1e-8) colsums[k] = 1 / sqrt(colsums[k]);
		else colsums[k] = 0;

	// Now normalize the matrix
	CS_INT p, j, n, *Ap, *Ai;
	CS_ENTRY *Ax;

	n = A->n;
	Ap = A->p;
	Ax = A->x;
	Ai = A->i;

	for (j = 0; j < n; j++) {
		for (p = Ap[j]; p < Ap[j + 1]; p++) {
			Ax[p] = colsums[j] * Ax[p] * colsums[Ai[p]];
		}
	}

	return 0;
}

// Maps a symmetric matrix A to (1/sqrt(D))*(A+D)*(1/sqrt(D))=1/2*(I+(1/sqrt(D))*A*(1/sqrt(D)))
cs *cs_W2DinvsqrtDpWDinvsqrt(cs *A) {
	if (!CS_CSC (A) || !A->x) return (NULL); /* check inputs */

	// First of all find the degree, by computing the sum of each column
	double *colsumsinvsqrt = cs_colsums(A);

	// Compute degree matrix
	cs *D = cs_Diag(A->n,colsumsinvsqrt);
	// Compute D+W
	cs *T  = cs_add(A,D,0.5,0.5);

	// Compute D^{-1/2}(D+W)D^{-1/2}

	// Compute D^{-1/2}
	for (int k = 0; k < A->m; k++)
		if (fabs(colsumsinvsqrt[k]) > 1e-8) colsumsinvsqrt[k] = 1 / sqrt(colsumsinvsqrt[k]);
		else colsumsinvsqrt[k] = 0;

	// Now normalize the matrix
	CS_INT p, j, n, *Ap, *Ai;
	CS_ENTRY *Ax;

	n = T->n;
	Ap = T->p;
	Ax = T->x;
	Ai = T->i;

	for (j = 0; j < n; j++) {
		for (p = Ap[j]; p < Ap[j + 1]; p++) {
			Ax[p] = colsumsinvsqrt[j] * Ax[p] * colsumsinvsqrt[Ai[p]];
		}
	}

	delete [] colsumsinvsqrt;
/*	colsumsinvsqrt = cs_colsums(A);

	cs *D = cs_Diag(A->n,colsumsinvsqrt);

	cs *T  = cs_add(A,D,0.5,0.5);
	cs_free( Id );*/

	return T;
}

// Constructs the identity matrix
cs* cs_Id(CS_INT n) {
	cs *T, *I;

	T = cs_spalloc(n, n, n, 1, 1); /* create triplet identity matrix */
	for (CS_INT i = 0; i < n; i++)
		cs_entry(T, i, i, 1);

	I = cs_compress(T);
	cs_spfree( T );

	return I;
}

// Constructs diagonal matrix with given entries
cs* cs_Diag(CS_INT n, double* diag){
	cs *D,*Dtmp;

	Dtmp = cs_spalloc(n,n,n,1,1);
	for (CS_INT i = 0; i < n; i++)
		cs_entry(Dtmp,i,i,diag[i]);

	D = cs_compress(Dtmp);
	cs_spfree(Dtmp);

	return D;
}

uGraph* ConvertCSparseToGraph(cs *A) {
	using namespace boost;

	if (!CS_CSC (A) || !A->x) return (NULL); /* check inputs */

	uGraph *G = new uGraph(A->n);
	CS_INT i,j;

	for( i=0; i<A->n; i++) {
		for( j = A->p[i]; j< A->p[i+1]-1; j++) {
			add_edge(A->i[j],i,*G);
		}
	}

    return G;
}

// Remove entries in the matrix that are repeated twice, replacing with the average
int cs_ave_dupl (cs *A)
{
    int i, j, p, q, nz = 0, n, m, *Ap, *Ai, *w ;
    double *Ax ;
    if (!CS_CSC (A)) return (0) ;               /* check inputs */
    m = A->m ; n = A->n ; Ap = A->p ; Ai = A->i ; Ax = A->x ;
    w = (int*)cs_malloc (m, sizeof (int)) ;           /* get workspace */
    if (!w) return (0) ;                        /* out of memory */
    for (i = 0 ; i < m ; i++) w [i] = -1 ;      /* row i not yet seen */
    for (j = 0 ; j < n ; j++)
    {
        q = nz ;                                /* column j will start at q */
        for (p = Ap [j] ; p < Ap [j+1] ; p++)
        {
            i = Ai [p] ;                        /* A(i,j) is nonzero */
            if (w [i] >= q)
            {
                Ax [w [i]] += Ax [p] ;          /* A(i,j) is a duplicate */
                Ax [w [i]] /= 2;
            }
            else
            {
                w [i] = nz ;                    /* record where row i occurs */
                Ai [nz] = i ;                   /* keep A(i,j) */
                Ax [nz++] = Ax [p] ;
            }
        }
        Ap [j] = q ;                            /* record start of column j */
    }
    Ap [n] = nz ;                               /* finalize A */
    cs_free (w) ;                               /* free workspace */
    return (cs_sprealloc (A, 0)) ;              /* remove extra space from A */
}

