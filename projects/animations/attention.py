from manim import *


class ScaledDotProduct(Scene):
    def construct(self) -> None:
        queries = MathTex('\\mathrm{Q}\;=\;\\{q_1,\;q_2,\;q_3\\}', font_size=24)
        keys = MathTex('\\mathrm{K}\;=\;\\{k_1,\;k_2,\;k_3\\}', font_size=24)
        values = MathTex('\\mathrm{V}\;=\;\\{v_1,\;v_2,\;v_3\\}', font_size=24)
        
        q_util = MathTex('\\mathrm{q}_i \\in \\mathbf{R}^{d_k}', font_size=24)
        k_util = MathTex('\\mathrm{k}_i \\in \\mathbf{R}^{d_k}', font_size=24)
        v_util = MathTex('\\mathrm{v}_i \\in \\mathbf{R}^{d_k}', font_size=24)

        q_matrix = MathTex(''' \\begin{bmatrix}
                q_{1,1} & q_{1,2} & \\dots & q_{1,d_k} \\\\
                q_{2,1} & q_{2,2} & \\dots & q_{2,d_k} \\\\
                q_{3,1} & q_{3,2} & \\dots & q_{3,d_k}
            \\end{bmatrix}''', font_size=24)

        kt_matrix = MathTex(''' \\begin{bmatrix}
                k_{1,1} & k_{2,1} & k_{3,1} \\\\
                k_{1,2} & k_{2,2} & k_{3,2} \\\\
                \\vdots & \\vdots & \\vdots \\\\
                k_{1,d_k} & k_{2,d_k} & k_{3,d_k}
            \\end{bmatrix}''', font_size=24)

        softmax_left = MathTex('softmax \\left(')
        softmax_right = MathTex('\\right)')

        equal_sign = MathTex('\;=\;')

        s_matrix = MathTex('''\\begin{bmatrix}
                s_{1,1} & s_{1,2} & s_{1,3} \\\\
                s_{2,1} & s_{2,2} & s_{2,2} \\\\
                s_{3,1} & s_{3,2} & s_{3,3}
            \\end{bmatrix}''', font_size=24)

        sij = MathTex('''
            S_{i,j}\;=\;\\sum_{l=1}^{d_{k}} Q_{i, l} . K_{l, j}
        ''', font_size=28)

        self.play(Write(queries))
        self.play(queries.animate.to_edge(UP + LEFT, buff=0.5))
        self.play(Write(keys))
        self.play(keys.animate.next_to(queries, DOWN))
        self.play(Write(values))    
        self.play(values.animate.next_to(keys, DOWN))
    
        q_util.next_to(queries, RIGHT)
        k_util.next_to(keys, RIGHT)
        v_util.next_to(values, RIGHT)

        self.play(Write(q_util))
        self.play(Write(k_util))
        self.play(Write(v_util))
        
        self.play(Write(q_matrix))
        self.play(q_matrix.animate.shift(2 * LEFT))
        kt_matrix.next_to(q_matrix, RIGHT)
        self.play(Write(kt_matrix))

        softmax_left.next_to(q_matrix, LEFT)
        softmax_right.next_to(kt_matrix, RIGHT)

        self.play(Write(softmax_left))
        self.play(Write(softmax_right))

        equal_sign.next_to(softmax_right, RIGHT)
        s_matrix.next_to(equal_sign, RIGHT)

        self.play(Write(equal_sign))
        self.play(Write(s_matrix))

        sij.next_to(q_matrix, DOWN)
        sij.shift(RIGHT)
        self.play(Write(sij))

        self.wait(5)