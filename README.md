<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License (MIT)][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/euctrl-pru/close_encounters">
    <img src="images/logo.svg" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Close Encounters</h3>

  <p align="center">
The close encounters Python package implements a fixed radius near neighbour search using (Uber's H3) cell techniques. Given a set of (aircraft) trajectories, it determines the optimal H3 resolution to work at given the fixed radius, interpolates and resamples (5s) the trajectories, adds Uber H3 indexes for each lat/lon at the optimal resolution and lastly applies the fixed radius near neighbour search cell techniques. 
    <br />
    <a href="https://github.com/euctrl-pru/close_encounters"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/euctrl-pru/close_encounters">View Repo</a>
    &middot;
    <a href="https://github.com/euctrl-pru/close_encounters/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/euctrl-pru/close_encounters/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!--[![Product Name Screen Shot][product-screenshot]](https://example.com)-->

### Built With

* [![Python][Python.org]][Python-url]
* [![PySpark][PySpark-logo]][PySpark-url]
* [![H3][H3-logo]][H3-url]
* [![NumPy][NumPy-logo]][NumPy-url]
* [![Pandas][Pandas-logo]][Pandas-url]

<!-- Badge Image References -->
[Python.org]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[PySpark-logo]: https://img.shields.io/badge/PySpark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white
[H3-logo]: https://img.shields.io/badge/H3-0066CC?style=for-the-badge&logo=hexo&logoColor=white
[NumPy-logo]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[Pandas-logo]: https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white

<!-- Badge URL References -->
[Python-url]: https://www.python.org/
[PySpark-url]: https://spark.apache.org/docs/latest/api/python/
[H3-url]: https://github.com/uber/h3-py
[NumPy-url]: https://numpy.org/
[Pandas-url]: https://pandas.pydata.org/


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get started make sure you have `python >= 3.12` and `pyspark>=3.2.3` installed.  

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* Python: Download [here](https://www.python.org/downloads/)
* Python dependencies:
  
```sh
pip install pyspark==3.2.3 
```

### Installation

1. Clone the repo
By HTTPS:
```sh
git clone https://github.com/euctrl-pru/close_encounters.git
```
Or by SSH:
```sh
git clone git@github.com:euctrl-pru/close_encounters.git 
```

3. PIP install cloned library locally
```sh
cd close_encounters && pip install -e .
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Once installed, you can run the code as follows: 

```python
from close_encounters.h3_half_disk import *

spark = SparkSession.builder \
    .appName("CloseEncountersH3") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "2") \
    .config("spark.sql.shuffle.partitions", "20") \
    .config("spark.rpc.message.maxSize", "256") \
    .getOrCreate()

encounters_df = CloseEncountersH3HalfDisk(coords_df, distance_nm = 5, FL_diff = 10, FL_min = 245, deltaT_min = 10, pnumb = 100, spark = spark)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Feature 1: Create a PIP Package
- [ ] Feature 2: Clean up and streamline code 

See the [open issues](https://github.com/euctrl-pru/close_encounters/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License (MIT). See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Quinten GOENS, Aviation Intelligence Unit at [EUROCONTROL](https://www.eurocontrol.int) 
* Email: [quinten.goens@eurocontrol.int](mailto:quinten.goens@eurocontrol.int) 

Project Link: [https://github.com/euctrl-pru/close_encounters](https://github.com/euctrl-pru/close_encounters)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Performance Review Commission (PRC)](https://www.eurocontrol.int/air-navigation-services-performance-review)
* [EUROCONTROL](https://www.eurocontrol.int)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/euctrl-pru/close_encounters.svg?style=for-the-badge
[contributors-url]: https://github.com/euctrl-pru/close_encounters/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/euctrl-pru/close_encounters.svg?style=for-the-badge
[forks-url]: https://github.com/euctrl-pru/close_encounters/network/members
[stars-shield]: https://img.shields.io/github/stars/euctrl-pru/close_encounters.svg?style=for-the-badge
[stars-url]: https://github.com/euctrl-pru/close_encounters/stargazers
[issues-shield]: https://img.shields.io/github/issues/euctrl-pru/close_encounters.svg?style=for-the-badge
[issues-url]: https://github.com/euctrl-pru/close_encounters/issues
[license-shield]: https://img.shields.io/github/license/euctrl-pru/close_encounters.svg?style=for-the-badge
[license-url]: https://github.com/euctrl-pru/close_encounters/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/quinten-goens
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
