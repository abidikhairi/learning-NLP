from manim import *

class TextEmbedding(Scene):
    def construct(self):
        input_text = Text("The cat sat on the mat", font_size=22)
        self.play(Write(input_text))
        self.play(input_text.animate.to_corner(UP))

        arrow1 = Arrow(UP, DOWN)
        arrow1.next_to(input_text, DOWN)
        self.play(Write(arrow1))

        tokenize_text = Text("Tokenize", font_size=32, color=BLUE_D)

        tokenize_text.next_to(arrow1, RIGHT)
        self.play(FadeIn(tokenize_text))
        
        tokenized_text = Tex("\{The, cat, sat, on, the, mat\}", font_size=26)
        tokenized_text.next_to(arrow1, DOWN)
        self.play(Write(tokenized_text))

        arrow2 = Arrow(UP, DOWN)
        arrow2.next_to(tokenized_text, DOWN)
        self.play(FadeIn(arrow2))

        embedding_func = MathTex('\mathrm{Embedding}_{\\theta}(\mathrm{word;\;\\theta})', font_size=36)
        embedding_func.next_to(arrow2, DOWN)
        self.play(FadeIn(embedding_func))


        x1 = MathTex('\\begin{bmatrix} 1.8 \\\\ -2.7 \\\\ 3.1 \end{bmatrix}', font_size=24)
        x2 = MathTex('\\begin{bmatrix} 8.4 \\\\ -0.5 \\\\ 4.3 \end{bmatrix}', font_size=24)
        x3 = MathTex('\\begin{bmatrix} 9.6 \\\\ 1.2 \\\\ 3.8 \end{bmatrix}', font_size=24)
        x4 = MathTex('\\begin{bmatrix} 4.7 \\\\ 5.1 \\\\ 0.2 \end{bmatrix}', font_size=24)
        x5 = MathTex('\\begin{bmatrix} 1.1 \\\\ -2.7 \\\\ 3.1 \end{bmatrix}', font_size=24)
        x6 = MathTex('\\begin{bmatrix} -2.8 \\\\ -6.7 \\\\ 7.1 \end{bmatrix}', font_size=24)

        x1.next_to(embedding_func, DOWN)
        x2.next_to(embedding_func, DOWN)

        x3.next_to(embedding_func, DOWN)
        x4.next_to(embedding_func, DOWN)

        x5.next_to(x4, RIGHT)
        x6.next_to(x5, RIGHT)

        self.play(Write(x1))
        self.play(x1.animate.shift(2.5 * LEFT))
        self.play(Write(x2))
        self.play(x2.animate.next_to(x1, RIGHT))
        self.play(Write(x3))
        self.play(x3.animate.next_to(x2, RIGHT))
        self.play(Write(x4))
        self.play(x4.animate.next_to(x3, RIGHT))
        self.play(Write(x5))
        self.play(x5.animate.next_to(x4, RIGHT))
        self.play(Write(x6))
        self.play(x6.animate.next_to(x5, RIGHT))
        
        
        word1 = Text("The", font_size=22)
        word2 = Text("cat", font_size=22)
        word3 = Text("sat", font_size=22)
        word4 = Text("on", font_size=22)
        word5 = Text("the", font_size=22)
        word6 = Text("mat", font_size=22)

        word1.next_to(x1, DOWN)
        word2.next_to(x2, DOWN)
        word3.next_to(x3, DOWN)
        word4.next_to(x4, DOWN)
        word5.next_to(x5, DOWN)
        word6.next_to(x6, DOWN)

        self.play(FadeIn(word1))
        self.play(FadeIn(word2))
        self.play(FadeIn(word3))
        self.play(FadeIn(word4))
        self.play(FadeIn(word5))
        self.play(FadeIn(word6))

        self.wait(4)