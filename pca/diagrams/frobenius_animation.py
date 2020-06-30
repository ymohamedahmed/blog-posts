from manimlib.imports import *

class FrobeniusScene(Scene):
    def transpose(self, mat):
        tr = np.where(mat.T==r"\dots", r"\vdots", mat.T)
        tr = np.where(mat.T==r"\vdots", r"\dots", tr)
        return tr

    def colour_row(self, mat, row, colour):
        cols = mat.mob_matrix.shape[1]
        for i in range(cols):
            mat.mob_matrix[row][i].set_color(colour)

    def colour_col(self, mat, col, colour):
        rows = mat.mob_matrix.shape[0]
        for i in range(rows):
            mat.mob_matrix[i][col].set_color(colour)

    def construct(self):
        # rect = Rectangle()  
        frobenius = VGroup(*TexMobject(r"Tr(\mathbf{A}\mathbf{A}^{\top})"))
        frobenius.shift(LEFT*6)
        frobenius.shift(UP*3.4)
        # self.add(frobenius)
        A_mat =  np.array([["A_{1,1}" , "A_{1,2}", r"\dots" , "A_{1,n}"],["A_{2,1}", "A_{2,2}", "\dots", r"A_{2,n}"] ,[r"\vdots", r"\vdots", r"\ddots", r"\vdots"],["A_{m,1}", "A_{m,2}", r"\dots", "A_{m,n}"]])
        # expansion = TexMobject(r" = Tr ", r"\left(\begin{bmatrix} \end{bmatrix} \begin{bmatrix} A_{1,1} & \dots & A_{m,1} \\ \vdots & \ddots & \\ A_{1,n} & \dots & A_{m,n} \end{bmatrix}\right)")
        first, sec = Matrix(A_mat), Matrix(self.transpose(A_mat))
        expansion = TexMobject(r" = Tr \big( ")
        expansion.next_to(frobenius, DOWN*3)
        first.next_to(expansion, RIGHT*0.7)
        sec.next_to(first, RIGHT)
        bracket = TexMobject(r"\big)")
        bracket.next_to(sec, RIGHT)
        # prod = VGroup(Matrix(A_mat), Matrix(A_mat.T))
        # prod.next_to(expansion)
        # self.add(expansion)
        self.play(Write(frobenius))
        # expansion.shift(RIGHT)
        self.play(Write(expansion))
        self.play(Write(first))
        self.play(Write(sec))
        self.play(Write(bracket))
        # self.play(Write(prod))
        # expansion.set_color_by_tex_to_color_map({
        #     "A_{1,1}": YELLOW
        # })
        self.wait()
        # self.add(expansion)
        # mat = Matrix([["A_{1,1}", "A_{2,1}"], ["A_{1,2}", "A_{2,2}"]])
        # self.add(mat)
        # first[0][1].set_color(YELLOW)
        # first.mob_matrix[1][1].set_color(YELLOW)
        self.colour_row(first, 0, RED)
        self.colour_col(sec, 0, RED)
        result = TexMobject(r"= A_{1,1}^2 + A_{1,2}^2 + .... + A_{1,n}^2")
        result.set_color(RED)
        result.next_to(first, DOWN)
        self.play(Write(result))
        self.wait() 

        self.colour_row(first, 1, YELLOW)
        self.colour_col(sec, 1, YELLOW)
        result2 = TexMobject(r"+ A_{2,1}^2 + A_{2,2}^2 + .... + A_{2,n}^2")
        result2.set_color(YELLOW)
        result2.next_to(result, DOWN)
        self.play(Write(result2))

        ellipsis = TexMobject(r"\dots")
        ellipsis.next_to(result2, DOWN)
        self.play(Write(ellipsis))

        self.wait()
        self.colour_row(first, 3, BLUE)
        self.colour_col(sec, 3, BLUE)
        result3 = TexMobject(r"+ A_{m,1}^2 + A_{m,2}^2 + .... + A_{m,n}^2")
        result3.set_color(BLUE)
        result3.next_to(ellipsis, DOWN)
        self.play(Write(result3))

        final_result = TexMobject(r"=\sum_{i,j}A_{i,j}^2 = ||A||_F^2")
        final_result.next_to(result3, RIGHT)
        self.play(Write(final_result))

        self.wait()