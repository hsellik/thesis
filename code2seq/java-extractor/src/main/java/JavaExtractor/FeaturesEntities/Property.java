package JavaExtractor.FeaturesEntities;

import JavaExtractor.Common.Common;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.expr.AssignExpr;
import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.IntegerLiteralExpr;
import com.github.javaparser.ast.expr.UnaryExpr;
import com.github.javaparser.ast.type.ClassOrInterfaceType;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Property {
    public static final HashSet<String> NumericalKeepValues = Stream.of("0", "1", "32", "64")
            .collect(Collectors.toCollection(HashSet::new));
    private static final Map<String, String> shortTypes = Collections.unmodifiableMap(new HashMap<String, String>() {
        /**
         *
         */
        private static final long serialVersionUID = 1L;

        {
            put("ArrayAccessExpr", "ArAc");
            put("ArrayBracketPair", "ArBr");
            put("ArrayCreationExpr", "ArCr");
            put("ArrayCreationLevel", "ArCrLvl");
            put("ArrayInitializerExpr", "ArIn");
            put("ArrayType", "ArTy");
            put("AssertStmt", "Asrt");
            put("AssignExpr:BINARY_AND", "AsAn");
            put("AssignExpr:ASSIGN", "As");
            put("AssignExpr:LEFT_SHIFT", "AsLS");
            put("AssignExpr:MINUS", "AsMi");
            put("AssignExpr:BINARY_OR", "AsOr");
            put("AssignExpr:PLUS", "AsP");
            put("AssignExpr:REMAINDER", "AsRe");
            put("AssignExpr:SIGNED_RIGHT_SHIFT", "AsRSS");
            put("AssignExpr:UNSIGNED_RIGHT_SHIFT", "AsRUS");
            put("AssignExpr:DIVIDE", "AsSl");
            put("AssignExpr:MULTIPLY", "AsSt");
            put("AssignExpr:XOR", "AsX");
            put("BinaryExpr:AND", "And");
            put("BinaryExpr:BINARY_AND", "BinAnd");
            put("BinaryExpr:BINARY_OR", "BinOr");
            put("BinaryExpr:DIVIDE", "Div");
            put("BinaryExpr:EQUALS", "Eq");
            put("BinaryExpr:GREATER", "Gt");
            put("BinaryExpr:GREATER_EQUALS", "Geq");
            put("BinaryExpr:LESS", "Ls");
            put("BinaryExpr:LESS_EQUALS", "Leq");
            put("BinaryExpr:LEFT_SHIFT", "LS");
            put("BinaryExpr:MINUS", "Minus");
            put("BinaryExpr:NOT_EQUALS", "Neq");
            put("BinaryExpr:OR", "Or");
            put("BinaryExpr:PLUS", "Plus");
            put("BinaryExpr:REMAINDER", "Mod");
            put("BinaryExpr:SIGNED_RIGHT_SHIFT", "RSS");
            put("BinaryExpr:UNSIGNED_RIGHT_SHIFT", "RUS");
            put("BinaryExpr:MULTIPLY", "Mul");
            put("BinaryExpr:XOR", "Xor");
            put("BlockStmt", "Bk");
            put("BooleanLiteralExpr", "BoolEx");
            put("CastExpr", "Cast");
            put("CatchClause", "Catch");
            put("CharLiteralExpr", "CharEx");
            put("ClassExpr", "ClsEx");
            put("ClassOrInterfaceDeclaration", "ClsD");
            put("ClassOrInterfaceType", "Cls");
            put("ConditionalExpr", "Cond");
            put("ConstructorDeclaration", "Ctor");
            put("DoStmt", "Do");
            put("DoubleLiteralExpr", "Dbl");
            put("EmptyMemberDeclaration", "Emp");
            put("EnclosedExpr", "Enc");
            put("ExplicitConstructorInvocationStmt", "ExpCtor");
            put("ExpressionStmt", "Ex");
            put("FieldAccessExpr", "Fld");
            put("FieldDeclaration", "FldDec");
            put("ForEachStmt", "Foreach");
            put("ForStmt", "For");
            put("IfStmt", "If");
            put("InitializerDeclaration", "Init");
            put("InstanceOfExpr", "InstanceOf");
            put("IntegerLiteralExpr", "IntEx");
            put("IntegerLiteralMinValueExpr", "IntMinEx");
            put("LabeledStmt", "Labeled");
            put("LambdaExpr", "Lambda");
            put("LongLiteralExpr", "LongEx");
            put("MarkerAnnotationExpr", "MarkerExpr");
            put("MemberValuePair", "Mvp");
            put("MethodCallExpr", "Cal");
            put("MethodDeclaration", "Mth");
            put("MethodReferenceExpr", "MethRef");
            put("NameExpr", "Nm");
            put("NormalAnnotationExpr", "NormEx");
            put("NullLiteralExpr", "Null");
            put("ObjectCreationExpr", "ObjEx");
            put("Parameter", "Prm");
            put("PrimitiveType", "Prim");
            put("QualifiedNameExpr", "Qua");
            put("ReturnStmt", "Ret");
            put("SingleMemberAnnotationExpr", "SMEx");
            put("StringLiteralExpr", "StrEx");
            put("SuperExpr", "SupEx");
            put("SwitchEntryStmt", "SwiEnt");
            put("SwitchStmt", "Switch");
            put("SynchronizedStmt", "Sync");
            put("ThisExpr", "This");
            put("ThrowStmt", "Thro");
            put("TryStmt", "Try");
            put("TypeDeclarationStmt", "TypeDec");
            put("TypeExpr", "Type");
            put("TypeParameter", "TypePar");
            put("UnaryExpr:BITWISE_COMPLEMENT", "Inverse");
            put("UnaryExpr:MINUS", "Neg");
            put("UnaryExpr:LOGICAL_COMPLEMENT", "Not");
            put("UnaryExpr:POSTFIX_DECREMENT", "PosDec");
            put("UnaryExpr:POSTFIX_INCREMENT", "PosInc");
            put("UnaryExpr:PLUS", "Pos");
            put("UnaryExpr:PREFIX_DECREMENT", "PreDec");
            put("UnaryExpr:PREFIX_INCREMENT", "PreInc");
            put("UnionType", "Unio");
            put("VariableDeclarationExpr", "VDE");
            put("VariableDeclarator", "VD");
            put("VariableDeclaratorId", "VDID");
            put("VoidType", "Void");
            put("WhileStmt", "While");
            put("WildcardType", "Wild");
            put("Modifier", "Mdfr");
            put("SimpleName", "SmplNm");

        }
    });
    private final String RawType;
    private String Type;
    private String SplitName;

    public Property(Node node, boolean isLeaf, boolean isGenericParent) {
        Class<?> nodeClass = node.getClass();
        RawType = Type = nodeClass.getSimpleName();
        if (node instanceof ClassOrInterfaceType && ((ClassOrInterfaceType) node).isBoxedType()) {
            Type = "PrimitiveType";
        }
        String operator = "";
        if (node instanceof BinaryExpr) {
            operator = ((BinaryExpr) node).getOperator().toString();
        } else if (node instanceof UnaryExpr) {
            operator = ((UnaryExpr) node).getOperator().toString();
        } else if (node instanceof AssignExpr) {
            operator = ((AssignExpr) node).getOperator().toString();
        }
        if (operator.length() > 0) {
            Type += ":" + operator;
        }

        String nameToSplit = node.toString();
        if (isGenericParent) {
            nameToSplit = ((ClassOrInterfaceType) node).getName().asString();
            if (isLeaf) {
                // if it is a generic parent which counts as a leaf, then when
                // it is participating in a path
                // as a parent, it should be GenericClass and not a simple
                // ClassOrInterfaceType.
                Type = "GenericClass";
            }
        }
        ArrayList<String> splitNameParts = Common.splitToSubtokens(nameToSplit);
        SplitName = String.join(Common.internalSeparator, splitNameParts);

        String name = Common.normalizeName(node.toString(), Common.BlankWord);
        if (name.length() > Common.c_MaxLabelLength) {
            name = name.substring(0, Common.c_MaxLabelLength);
        } else if (node instanceof ClassOrInterfaceType && ((ClassOrInterfaceType) node).isBoxedType()) {
            name = ((ClassOrInterfaceType) node).toUnboxedType().toString();
        }

//        if (Common.isMethod(node, Type)) {
//            name = SplitName = Common.methodName;
//        }

        if (SplitName.length() == 0) {
            SplitName = name;
//            // Code to replace numbers with <NUM>
//            if (node instanceof IntegerLiteralExpr && !NumericalKeepValues.contains(SplitName)) {
//                // This is a numeric literal, but not in our white list
//                SplitName = "<NUM>";
//            }
        }
    }

    public String getRawType() {
        return RawType;
    }

    public String getType() {
        return Type;
    }

    public String getType(boolean shorten) {
        if (shorten) {
            return shortTypes.getOrDefault(Type, Type);
        } else {
            return Type;
        }
    }

    public String getName() {
        return SplitName;
    }
}
