import React, {Component} from 'react';
import './header.css'
// import { connect } from "react-redux";
// import { bindActionCreators } from "redux";
// import * as headerActions from "../../store/header/actions";
export default class header extends Component {
    // constructor(props) {
    //     super(props);
    //     this.state = {};
    // }
    render() {
        return (
            <header id="header">
                <a className="logo" href="index.html">OffSide</a>
                <nav>
                    <div className="image-logo-container">
                        <img className="image-logo" src={require('./../../images/delft-logo.png')} alt={"delft logo"} />
                    </div>

                </nav>
            </header>
        );
    }
}
// export default connect(
//     ({ header }) => ({ ...header }),
//     dispatch => bindActionCreators({ ...headerActions }, dispatch)
//   )( header );