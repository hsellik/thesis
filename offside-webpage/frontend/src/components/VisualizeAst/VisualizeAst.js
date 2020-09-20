import React, {Component} from 'react';
import './VisualizeAst.css'
import Tree from 'react-d3-tree'
import AceEditor from "react-ace";
import ace from 'ace-builds'

import "ace-builds/src-noconflict/mode-java";
import "ace-builds/src-noconflict/theme-xcode";

const {Range} = ace.require("ace/range");

const svgSquare = {
    shape: 'rect',
    shapeProps: {
        width: 130,
        height: 70,
        x: -65,
        y: 0
    },
    textLayout: {textAnchor: "start", x: -70, y: -10, transform: undefined }
};

export default class VisualizeAst extends Component {
    constructor(props) {
        super(props);
        this.state = {
            displayAlert: false,
            isLoading: false,
            resultBtnClass: "button inactive",
            resultBtnText: "Result",
            resultConfidence: 0.00,
            mouseOverMarkers: [],
            code: `public String[] reverseArray(final String[] array) {
  final String[] newArray = new String[array.length];
  for (int index = 0; index < array.length; index++) {
    newArray[array.length - index - 1] = array[index];
  }
  return newArray;
}`,
            myTreeData3: {
                "paths": [
                    {
                        "source": "public",
                        "sourceId": 1,
                        "target": "METHOD_NAME",
                        "targetId": 2,
                        "shortPath": "Mdfr0|Mth|SmplNm1",
                        "longPath": "(Modifier0)^MethodDeclaration_(SimpleName1)"
                    },
                    {
                        "source": "public",
                        "sourceId": 1,
                        "target": "void",
                        "targetId": 3,
                        "shortPath": "Mdfr0|Mth|Void2",
                        "longPath": "(Modifier0)^MethodDeclaration_(VoidType2)"
                    },
                ],
                "ast": {
                    "id": 0,
                    "range": {"begin":{"line":1,"column":20},"end":{"line":8,"column":1}},
                    "name": "MethodDeclaration",
                    "type": "MethodDeclaration",
                    nodeSvgShape: {
                        shape: 'rect',
                        shapeProps: {
                            width: 120,
                            height: 60,
                            x: -60,
                            y: 0,
                            stroke: 'green',
                            strokeWidth: 4
                        }
                    },
                    "children": [
                        {
                            "id": 1,
                            "range": {"begin":{"line":1,"column":20},"end":{"line":1,"column":25}},
                            "type": "Modifier",
                            "name": "public"
                        },
                        {
                            "id": 2,
                            "range": {"begin":{"line":1,"column":32},"end":{"line":1,"column":42}},
                            "type": "SimpleName",
                            "name": "METHOD_NAME"
                        },
                        {
                            "id": 3,
                            "range": {"begin":{"line":1,"column":27},"end":{"line":1,"column":30}},
                            "type": "VoidType",
                            "name": "void"
                        },
                        {
                            "id": 4,
                            "range": {"begin":{"line":1,"column":46},"end":{"line":8,"column":1}},
                            "type": "BlockStmt",
                            "children": [
                                {
                                    "id": 5,
                                    "range": {"begin":{"line":2,"column":5},"end":{"line":2,"column":19}},
                                    "type": "ExpressionStmt",
                                    "children": [
                                        {
                                            "id": 6,
                                            "range": {"begin":{"line":2,"column":5},"end":{"line":2,"column":18}},
                                            "type": "VariableDeclarationExpr",
                                            "children": [
                                                {
                                                    "id": 7,
                                                    "range": {"begin":{"line":2,"column":12},"end":{"line":2,"column":18}},
                                                    "type": "VariableDeclarator",
                                                    "children": [
                                                        {
                                                            "id": 8,
                                                            "range": {"begin":{"line":2,"column":5},"end":{"line":2,"column":10}},
                                                            "type": "ClassOrInterfaceType",
                                                            "children": [
                                                                {
                                                                    "id": 9,
                                                                    "range": {"begin":{"line":2,"column":5},"end":{"line":2,"column":10}},
                                                                    "type": "SimpleName",
                                                                    "name": "string"
                                                                }
                                                            ]
                                                        },
                                                        {
                                                            "id": 10,
                                                            "range": {"begin":{"line":2,"column":12},"end":{"line":2,"column":12}},
                                                            "type": "SimpleName",
                                                            "name": "a"
                                                        },
                                                        {
                                                            "id": 11,
                                                            "range": {"begin":{"line":2,"column":16},"end":{"line":2,"column":18}},
                                                            "type": "StringLiteralExpr",
                                                            "name": "a"
                                                        }
                                                    ]
                                                }
                                            ]
                                        }
                                    ]
                                },
                                {
                                    "id": 12,
                                    "range": {"begin":{"line":3,"column":5},"end":{"line":5,"column":5}},
                                    "type": "IfStmt",
                                    "children": [
                                        {
                                            "id": 13,
                                            "range": {"begin":{"line":3,"column":9},"end":{"line":3,"column":17}},
                                            "type": "BinaryExpr",
                                            "children": [
                                                {
                                                    "id": 14,
                                                    "range": {"begin":{"line":3,"column":9},"end":{"line":3,"column":9}},
                                                    "type": "NameExpr",
                                                    "children": [
                                                        {
                                                            "id": 15,
                                                            "range": {"begin":{"line":3,"column":9},"end":{"line":3,"column":9}},
                                                            "type": "SimpleName",
                                                            "name": "a"
                                                        }
                                                    ]
                                                },
                                                {
                                                    "id": 16,
                                                    "range": {"begin":{"line":3,"column":14},"end":{"line":3,"column":17}},
                                                    "type": "NullLiteralExpr",
                                                    "name": "null"
                                                }
                                            ]
                                        },
                                        {
                                            "id": 17,
                                            "range": {"begin":{"line":3,"column":20},"end":{"line":5,"column":5}},
                                            "type": "BlockStmt",
                                            "children": [
                                                {
                                                    "id": 18,
                                                    "range": {"begin":{"line":4,"column":9},"end":{"line":4,"column":38}},
                                                    "type": "ExpressionStmt",
                                                    "children": [
                                                        {
                                                            "id": 19,
                                                            "range": {"begin":{"line":4,"column":9},"end":{"line":4,"column":37}},
                                                            "type": "MethodCallExpr",
                                                            "children": [
                                                                {
                                                                    "id": 20,
                                                                    "range": {"begin":{"line":4,"column":9},"end":{"line":4,"column":18}},
                                                                    "type": "FieldAccessExpr",
                                                                    "children": [
                                                                        {
                                                                            "id": 21,
                                                                            "range": {"begin":{"line":4,"column":9},"end":{"line":4,"column":14}},
                                                                            "type": "NameExpr",
                                                                            "children": [
                                                                                {
                                                                                    "id": 22,
                                                                                    "range": {"begin":{"line":4,"column":9},"end":{"line":4,"column":14}},
                                                                                    "type": "SimpleName",
                                                                                    "name": "system"
                                                                                }
                                                                            ]
                                                                        },
                                                                        {
                                                                            "id": 23,
                                                                            "range": {"begin":{"line":4,"column":16},"end":{"line":4,"column":18}},
                                                                            "type": "SimpleName",
                                                                            "name": "out"
                                                                        }
                                                                    ]
                                                                },
                                                                {
                                                                    "id": 24,
                                                                    "range": {"begin":{"line":4,"column":20},"end":{"line":4,"column":26}},
                                                                    "type": "SimpleName",
                                                                    "name": "println"
                                                                },
                                                                {
                                                                    "id": 25,
                                                                    "range": {"begin":{"line":4,"column":28},"end":{"line":4,"column":36}},
                                                                    "type": "StringLiteralExpr",
                                                                    "name": "asdfasd"
                                                                }
                                                            ]
                                                        }
                                                    ]
                                                }
                                            ]
                                        }
                                    ]
                                },
                                {
                                    "id": 26,
                                    "range": {"begin":{"line":5,"column":7},"end":{"line":7,"column":5}},
                                    "type": "IfStmt",
                                    "children": [
                                        {
                                            "id": 27,
                                            "range": {"begin":{"line":5,"column":11},"end":{"line":5,"column":16}},
                                            "type": "BinaryExpr",
                                            "children": [
                                                {
                                                    "id": 28,
                                                    "range": {"begin":{"line":5,"column":11},"end":{"line":5,"column":11}},
                                                    "type": "IntegerLiteralExpr",
                                                    "name": "\u003cNUM\u003e"
                                                },
                                                {
                                                    "id": 29,
                                                    "range": {"begin":{"line":5,"column":15},"end":{"line":5,"column":16}},
                                                    "type": "IntegerLiteralExpr",
                                                    "name": "\u003cNUM\u003e"
                                                }
                                            ]
                                        },
                                        {
                                            "id": 30,
                                            "range": {"begin":{"line":5,"column":19},"end":{"line":7,"column":5}},
                                            "type": "BlockStmt",
                                            "children": [
                                                {
                                                    "id": 31,
                                                    "range": {"begin":{"line":6,"column":9},"end":{"line":6,"column":34}},
                                                    "type": "ExpressionStmt",
                                                    "children": [
                                                        {
                                                            "id": 32,
                                                            "range": {"begin":{"line":6,"column":9},"end":{"line":6,"column":33}},
                                                            "type": "MethodCallExpr",
                                                            "children": [
                                                                {
                                                                    "id": 33,
                                                                    "range": {"begin":{"line":6,"column":9},"end":{"line":6,"column":18}},
                                                                    "type": "FieldAccessExpr",
                                                                    "children": [
                                                                        {
                                                                            "id": 34,
                                                                            "range": {"begin":{"line":6,"column":9},"end":{"line":6,"column":14}},
                                                                            "type": "NameExpr",
                                                                            "children": [
                                                                                {
                                                                                    "id": 35,
                                                                                    "range": {"begin":{"line":6,"column":9},"end":{"line":6,"column":14}},
                                                                                    "type": "SimpleName",
                                                                                    "name": "system"
                                                                                }
                                                                            ]
                                                                        },
                                                                        {
                                                                            "id": 36,
                                                                            "range": {"begin":{"line":6,"column":16},"end":{"line":6,"column":18}},
                                                                            "type": "SimpleName",
                                                                            "name": "out"
                                                                        }
                                                                    ]
                                                                },
                                                                {
                                                                    "id": 37,
                                                                    "range": {"begin":{"line":6,"column":20},"end":{"line":6,"column":26}},
                                                                    "type": "SimpleName",
                                                                    "name": "println"
                                                                },
                                                                {
                                                                    "id": 38,
                                                                    "range": {"begin":{"line":6,"column":28},"end":{"line":6,"column":32}},
                                                                    "type": "StringLiteralExpr",
                                                                    "name": "bee"
                                                                }
                                                            ]
                                                        }
                                                    ]
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            },
            textLayout: {textAnchor: "middle", x: 0, y: 30, transform: undefined },
            treeStyle: {
                links: {stroke: 'black', strokeWidth: 2,},
                    nodes: {
                        node: {
                        circle: {fill: 'lightgray', stroke: 'black', strokeWidth: 2}
                        },
                        leafNode: {
                            circle: {fill: 'white', stroke: 'black', strokeWidth: 2},
                            name: {stroke: 'black'},
                            attributes: {stroke: 'blue'}
                            },
                            },
                            }}
    }

    componentDidMount() {
        const dimensions = this.treeContainer.getBoundingClientRect();
        this.setState({
            translate: {
                x: dimensions.width / 2,
                y: dimensions.height / 2
            }
        });
        this.inside(this.state.myTreeData3.ast);
        this.createRequest();
    }

    inside(ast) {
        var key;
        for (key in ast) {
            if (Array.isArray(ast[key])) {
                this.inside(ast[key]);
            }
        }
    }

    onMouseOver(nodeData, evt) {
        if (typeof nodeData !== 'undefined') {
            let range = nodeData.range;
            this.setState({mouseOverMarkers : this.state.mouseOverMarkers.concat(this.highlight(range.begin.line, range.begin.column, range.end.line, range.end.column))})
        }
    }

    onMouseOut(nodeData, evt) {
        for(var i = this.state.mouseOverMarkers -1; i >= 0 ; i--){
            this.removeHighlight(this.state.mouseOverMarkers[i]);
            this.state.mouseOverMarkers.splice(i, 1);
        }
    }

    onAceChange(code, evt) {
        this.setState({code: code})
    }

    removeHighlight(markerId) {
        const reactAceComponent = this.refs.aceEditor;
        if (typeof reactAceComponent !== 'undefined') {
            const editor = reactAceComponent.editor;
            editor.session.removeMarker(markerId);
        }
    }

    highlight(beginLine,beginColumn,endLine,endColumn) {
        beginLine--;
        beginColumn--;
        endLine--;
        const reactAceComponent = this.refs.aceEditor;
        if (typeof reactAceComponent !== 'undefined') {
            const editor = reactAceComponent.editor;
            return editor.session.addMarker(new Range(beginLine,beginColumn,endLine,endColumn), "marker7", "text", false);
        }
    }

    createRequest() {
        (async () => {
            this.setState({isLoading : true});
            try {
                const rawResponse = await fetch('/predict/', {
                    method: 'POST',
                    headers: {
                        'Accept': 'application/json',
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({code: this.state.code})
                });
                const content = await rawResponse.json();
                this.setState({myTreeData3: content});
                if (parseFloat(content.bug) > 0.5) {
                    this.setState({resultBtnText: "Defective"})
                    this.setState({resultBtnClass: "button defective"})
                    this.setState( {resultConfidence : parseFloat(content.bug).toPrecision(2)})
                } else {
                    this.setState({resultBtnText: "No Bug"})
                    this.setState({resultBtnClass: "button non-defective"})
                    this.setState( {resultConfidence : parseFloat(content.nobug).toPrecision(2)})
                }
                this.setState({displayAlert: false});
            } catch (e) {
                this.setState({displayAlert: true});
            }
            this.setState({isLoading : false});

        })();
    }

    render() {
        return (
            <div className="container-fluid smaller-container">
                {(() => {
                    if (this.state.displayAlert) {
                        return <div className="alert alert-danger">
                            Please make sure the Java method has at least one binary operator
                            (>, >=, etc.) and the Java syntax is correct.
                        </div>
                    } else {
                        return
                    }
                })()}
                <div className="row">
                    <div className="col-md-4 pt-4">
                        <h3>Code</h3>
                        <AceEditor
                            ref={"aceEditor"}
                            placeholder={"Java Editor"}
                            mode={"java"}
                            theme={"xcode"}
                            name={"ace-editor"}
                            className={"border"}
                            width={"100%"}
                            height={"67vh"}
                            onLoad={this.onLoad}
                            onChange={this.onAceChange.bind(this)}
                            fontSize={12}
                            showPrintMargin={true}
                            showGutter={false}
                            highlightActiveLine={false}
                            setOptions={{
                                showLineNumbers: false,
                                tabSize: 2,
                            }}
                            value={this.state.code}/>
                    </div>

                    <div className="col-md-1 my-auto pt-4">
                        <div className="row justify-content-center">
                            {(() => {
                                if (this.state.isLoading) {
                                    return <div className="col-xs-6 mb-5">
                                               <div className="loading-wrapper">
                                                    <div className="loading circled crcld4">
                                                        <span className="circle"></span>
                                                    </div>
                                                </div>
                                           </div>
                                } else {
                                    return <div className="col-xs-6 mt-2">
                                              <a onClick={this.createRequest.bind(this)} className="button  primary">Detect</a>
                                           </div>;
                                }
                            })()}
                        </div>
                        <br/>
                        <div className="row justify-content-center mt-2">
                            <div className="col-xs-6">
                                <a id="btn-result" className={this.state.resultBtnClass}>{this.state.resultBtnText}</a>
                            </div>
                        </div>
                        <div className="row justify-content-center">
                            <div className="col-xs-6">
                                <h4>Confidence</h4>
                            </div>
                        </div>
                        <div className="row justify-content-center">
                            <div className="col-xs-6">
                                <a id="btn-result" className={this.state.resultBtnClass}>{this.state.resultConfidence}</a>
                            </div>
                        </div>
                    </div>

                    {/*<Tree /> will fill width/height of its container; in this case `#treeWrapper`*/}
                    <div className="col-md-7 pr-3 rd3t-tree-col pt-4">
                        <h3>AST</h3>
                        <div id="treeWrapper" className="rd3t-tree-container border" ref={tc => (this.treeContainer = tc)}>
                            <Tree
                                onMouseOver={this.onMouseOver.bind(this)}
                                onMouseOut={this.onMouseOut.bind(this)}
                                data={this.state.myTreeData3.ast}
                                translate={this.state.translate}
                                zoom={0.3}
                                styles={this.state.treeStyle}
                                textLayout={this.state.textLayout}
                                nodeSvgShape={svgSquare}
                                initialDepth={3}
                                orientation={'vertical'}
                                allowForeignObjects={true}
                                nodeLabelComponent={{
                                    render: <NodeLabel />,
                                    foreignObjectWrapper: {
                                        y: 0,
                                        x: -58
                                    }
                                }}
                            />
                        </div>
                    </div>
                </div>
            </div>

        );
    }
}

class NodeLabel extends React.PureComponent {
    render() {
        const {nodeData} = this.props
        return (
            <div className="node-wrapper" style={{height: svgSquare.shapeProps.height, color: "red"}}>
                <p className="node-content">{nodeData.name}</p>
            </div>
        )
    }
}