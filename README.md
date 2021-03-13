# Personal Website

Personal website. Basic steps to add a new blog post.

1. Save Jupyter notebook as markdown, which will output a zipped folder.
2. Copy the markdown file inside output folder into `./_posts/`, renaming file following the convention `[YYYY]-[MM]-[DD]-[TITLE].markdown`.
3. Prepend a header of the following format in markdown file

> ---
> layout: post
> title:  "[TITLE]"
> date:   [YYYY]-[MM]-[DD]
> categories: [CATEGORY]
> ---

4. The image outputs from Jupyter notebook will be saved in output folder from step 1. Move those images to `./assets/img/[TITLE]_files/[IMAGE_NAME].png`. In the markdown file, reference each image as

```
![png]({{site.baseurl}}/assets/img/[TITLE]_files/[IMAGE_NAME].png)
```

5. Run `bundle exec jekyll serve` and visit server address returned by the command in your local browser. This creates the necessary files in `./_site/` (e.g. html file of markdown) so that GitHub can host the blog post as a webpage.

6. Run `./publish.sh` to push changes to GitHub. Now visit link of website and the new blog post should be at `[BASE_URL]/[CATEGORY]/[YYYY]/[MM]/[DD]/[TITLE].html`. 

