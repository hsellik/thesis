import React from 'react';
import 'bootstrap/dist/css/bootstrap.css';
import './App.css';
import VisualizeAst from "./components/VisualizeAst";
import Header from "./components/header";
// template imports
import './assets/css/font-awesome.min.css'
import './assets/css/main.css'


function App() {
    return (
        <div>
            <div className="App">
                <Header  />

                {/*<!-- Banner -->*/}
                <section id="banner">
                    <div className="inner">
                        <h1>Learning Off-By-One Mistakes</h1>
                        <p>An Empirical Study on Different Deep Learning Models.</p>
                        <div className="row justify-content-center">
                            <div className="col-1-xsmall my-auto">
                                <a target="_blank" rel="noopener noreferrer" href="https://repository.tudelft.nl/islandora/object/uuid%3A1fe836a3-1874-497e-b05e-666588168717">
                                    <img alt="paper" src={require('./images/paper.png')} />
                                </a>
                            </div>
                            <div className="col-3-xsmall pl-1 my-auto">
                                <span><u><a target="_blank" rel="noopener noreferrer" href="https://repository.tudelft.nl/islandora/object/uuid%3A1fe836a3-1874-497e-b05e-666588168717">Paper</a></u></span>
                            </div>
                            <div className="col-1-xsmall my-auto">
                                <a target="_blank" rel="noopener noreferrer" href="https://github.com/hsellik/thesis">
                                    <img alt="github" src={require('./images/github.png')} />
                                </a>
                            </div>
                            <div className="col-3-xsmall pl-1 my-auto">
                                <span><u><a target="_blank" rel="noopener noreferrer" href="https://github.com/hsellik/thesis">GitHub</a></u></span>
                            </div>
                        </div>
                    </div>
                    <video  loop autoPlay muted>
                        <source src={require('./images/mekelpark.mp4')} type="video/mp4" />Your browser does not support the video tag. I suggest you upgrade your browser.
                    </video>
                </section>

                <VisualizeAst />

                {/*<!-- CTA -->*/}
                <section id="cta" className="wrapper">
                    <div className="inner">
                        <h2>About</h2>
                        <p> This is a web page to demonstrate the working principles described in the Master thesis.</p>
                    </div>
                </section>

                {/*<!-- Testimonials -->*/}
                <section className="wrapper">
                    <div className="inner">
                        <header className="special">
                            <h2>Authors</h2>
                        </header>
                        <div className="highlights">
                            <section className="author-left">
                                <div className="content">
                                    <header>
                                        <img alt="profile" src={require('./images/hendrig-profile.png')} className="image--circle" />
                                        <h3>Hendrig Sellik</h3>
                                    </header>
                                    <p>Delft University of Technology</p>
                                </div>
                            </section>
                            <section className="author-right">
                                <div className="content">
                                    <header>
                                        <img alt="profile" src={require('./images/mauricio-profile.jpg')}
                                             className="image--circle"/>
                                        <h3>Maurício Aniche</h3>
                                    </header>
                                    <p>Delft University of Technology</p>
                                </div>
                            </section>
                        </div>
                    </div>
                </section>

                {/*<!-- Footer -->*/}
                <footer id="footer">
                    <div className="copyright">
                        &copy; Hendrig Sellik. Photos: <a href="https://commons.wikimedia.org/wiki/File:Mekel_Park_-_Campus_Delft_University_of_Technology_01.jpg">Wikimedia</a>, <a href="https://unsplash.co/">Unsplash</a>. Video: <a href="https://www.youtube.com/user/Mecanoo84">Mecanoo</a>.
                    </div>
                </footer>
            </div>
        </div>
    );
}

export default App;
