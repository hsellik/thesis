package Tokenizer.Visitors;

import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.expr.ArrayInitializerExpr;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.ast.stmt.*;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.ReferenceType;
import com.github.javaparser.ast.type.UnionType;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.util.ArrayList;

/**
 * A quote from Bugram paper by Song Wang et al. Section 3.1
 * This visitor tries to follow those guidelines
 * <p>
 * "We focus on method calls and control flow which are:
 * method calls, constructors, and initializers; if/else branches;
 * for/do/while/foreach loops; break/continue statements;
 * try/catch/finally blocks; return statements; synchronized blocks;
 * switch statements and case/default branches, and assert statements.
 * <p>
 * A method call methodA() is resolved to its fully qualified
 * name org.example.Foo.methodA() to prevent unrelated
 * methods with an identical name from being grouped together. In
 * addition, the type of exception in the catch clauses are considered
 * as they provide important context information to help us infer
 * more accurate contextual information of method sequences"
 */
public class BugramVisitor extends VoidVisitorAdapter<Object> {
    private final ArrayList<String> tokens;
    private final String END_PREDIX = "END_";

    public BugramVisitor(ArrayList<String> tokens) {
        this.tokens = tokens;
    }

    @Override
    public void visit(MethodCallExpr n, Object arg) {
        tokens.add(n.getNameAsString());
        //System.out.println(n.getNameAsString());
        super.visit(n, arg);
    }

    // Constructordeclaration never found in a method???
    @Override
    public void visit(ConstructorDeclaration n, Object arg) {
        tokens.add(n.getClass().getSimpleName());

        //System.out.println(n.getClass().getSimpleName());
        super.visit(n, arg);
    }

    @Override
    public void visit(ObjectCreationExpr n, Object arg) {
        tokens.add(n.getClass().getSimpleName());

        //System.out.println(n.getClass().getSimpleName());
        super.visit(n, arg);
    }

    @Override
    public void visit(ThrowStmt n, Object arg) {
        tokens.add(n.getClass().getSimpleName());

        //System.out.println(n.getClass().getSimpleName());
        super.visit(n, arg);
    }

    @Override
    public void visit(ArrayInitializerExpr n, Object arg) {
        tokens.add(n.getClass().getSimpleName());
        //System.out.println(n.getClass().getSimpleName());
        super.visit(n, arg);
    }

    // This handles "if" and "else if", but not "else"
    @Override
    public void visit(IfStmt n, Object arg) {
        tokens.add(n.getClass().getSimpleName());
        //System.out.println(n.getClass().getSimpleName());
        super.visit(n, arg);
    }

    @Override
    public void visit(ForStmt n, Object arg) {
        tokens.add(n.getClass().getSimpleName());
        //System.out.println(n.getClass().getSimpleName());
        super.visit(n, arg);
        tokens.add(END_PREDIX + n.getClass().getSimpleName());
        //System.out.println(END_PREDIX + n.getClass().getSimpleName());
    }

    @Override
    public void visit(DoStmt n, Object arg) {
        tokens.add(n.getClass().getSimpleName());
        //System.out.println(n.getClass().getSimpleName());
        super.visit(n, arg);
    }

    @Override
    public void visit(WhileStmt n, Object arg) {
        tokens.add(n.getClass().getSimpleName());
        //System.out.println(n.getClass().getSimpleName());
        super.visit(n, arg);
    }

    @Override
    public void visit(ForEachStmt n, Object arg) {
        tokens.add(n.getClass().getSimpleName());
        //System.out.println(n.getClass().getSimpleName());
        super.visit(n, arg);
        tokens.add(END_PREDIX + n.getClass().getSimpleName());
        //System.out.println(END_PREDIX + n.getClass().getSimpleName());
    }

    @Override
    public void visit(BreakStmt n, Object arg) {
        tokens.add(n.getClass().getSimpleName());
        //System.out.println(n.getClass().getSimpleName());
        super.visit(n, arg);
    }

    @Override
    public void visit(ContinueStmt n, Object arg) {
        tokens.add(n.getClass().getSimpleName());
        //System.out.println(n.getClass().getSimpleName());
        super.visit(n, arg);
    }

    @Override
    public void visit(TryStmt n, Object arg) {
        tokens.add(n.getClass().getSimpleName());
        //System.out.println(n.getClass().getSimpleName());
        super.visit(n, arg);
        tokens.add(END_PREDIX + n.getClass().getSimpleName());
        //System.out.println(END_PREDIX + n.getClass().getSimpleName());
    }

    @Override
    public void visit(CatchClause n, Object arg) {
        Parameter parameter = n.getParameter();
        ArrayList<String> exceptionsCaught = new ArrayList<>();

        if (parameter.getType() instanceof ClassOrInterfaceType) {
            ClassOrInterfaceType type = (ClassOrInterfaceType) parameter.getType();
            exceptionsCaught.add(type.getNameAsString());
        } else {
            UnionType parameterType = (UnionType) parameter.getType();
            for (ReferenceType element : parameterType.getElements()) {
                ClassOrInterfaceType castedElement = (ClassOrInterfaceType) element;
                exceptionsCaught.add(castedElement.getNameAsString());
            }
        }
        exceptionsCaught.sort(String.CASE_INSENSITIVE_ORDER);

        for (String exception : exceptionsCaught) {
            tokens.add(n.getClass().getSimpleName() + "." + exception);
            //System.out.println(n.getClass().getSimpleName() + "." + exception);
        }

        super.visit(n, arg);
    }

    //finally block skipped, no default visitor found in JavaParser

    // Made for detecting "else" statements in control flow
    @Override
    public void visit(BlockStmt n, Object arg) {
        if (n.getParentNode().isPresent() && n.getParentNode().get() instanceof IfStmt) {
            IfStmt parentIfStmnt = (IfStmt) n.getParentNode().get();
            Statement parentElseStmnt = parentIfStmnt.getElseStmt().isPresent() ? parentIfStmnt.getElseStmt().get() : null;
            if (n.equals(parentElseStmnt)) {
                tokens.add("elseStmt");
                //System.out.println("elseStmt");
            }
        }
        super.visit(n, arg);
    }

    @Override
    public void visit(ReturnStmt n, Object arg) {
        tokens.add(n.getClass().getSimpleName());
        //System.out.println(n.getClass().getSimpleName());
        super.visit(n, arg);
    }

    @Override
    public void visit(SynchronizedStmt n, Object arg) {
        tokens.add(n.getClass().getSimpleName());
        //System.out.println(n.getClass().getSimpleName());
        super.visit(n, arg);
        tokens.add(END_PREDIX + n.getClass().getSimpleName());
        //System.out.println(END_PREDIX + n.getClass().getSimpleName());
    }

    @Override
    public void visit(SwitchStmt n, Object arg) {
        tokens.add(n.getClass().getSimpleName());
        //System.out.println(n.getClass().getSimpleName());
        super.visit(n, arg);
        tokens.add(END_PREDIX + n.getClass().getSimpleName());
        //System.out.println(END_PREDIX + n.getClass().getSimpleName());
    }

    // Case/default branch skipped, no default visitor found in JavaParser

    @Override
    public void visit(AssertStmt n, Object arg) {
        tokens.add(n.getClass().getSimpleName());
        //System.out.println(n.getClass().getSimpleName());
        super.visit(n, arg);
    }


}
