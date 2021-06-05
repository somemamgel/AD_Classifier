server_url = 'http://127.0.0.1:9000/api/prediction';  // 默认请求地址

// var res_text_dict = {
// };


// 获取上传图片的路径
function getFileUri(obj) {
    let uri;
    uri = window.URL.createObjectURL(obj.files.item(0));
    return uri;
}

// Vue 处理请求
const app = Vue.createApp({
    data() {
        return {
            filename: '',
            image_path: '',
            ad_value: null,
            mci_value: null,
            nc_value: null,
        }
    },
    methods: {
        handleFileUpload(event) {
            let ftype = this.$refs.file.files[0].type;
            let fpath = getFileUri(this.$refs.file);
            if (true/*ftype == 'image/jpeg' || ftype == 'image/png'*/) {
                this.filename = this.$refs.file.files[0].name;
                this.file = this.$refs.file.files[0];
                this.image_path = fpath;
                // readFile(this.$refs.file.files[0]);
            }
            else {
                alert('请选择 .nii .nii.gz 文件')
                this.filename = this.image_path = '';
            }
        },
        handleSubmitRequests(event) {
            if (this.file) {
                let file = this.file;
                let param = new FormData();
                param.append('image', file, file.name);
                axios({
                    method: 'post',
                    url: server_url,
                    data: param,
                    headers: {}
                }).then((response) => {
                    let result = response.data;
                    if (result[0] === 'prob' && Object.keys(result[1]).length == 3) {
                        console.log("data received from api!");
                        let r = result[1];
                        this.ad_value = (r['ad'] * 100).toFixed(4) + '%';
                        this.mci_value = (r['mci'] * 100).toFixed(4) + '%';
                        this.nc_value = (r['nc'] * 100).toFixed(4) + '%';
                        // let max = 0;
                        // let index;
                        // for (let i in r) {
                        //     if (r[i] > max) {
                        //         index = i;
                        //         max = r[i];
                        //     }
                        // }
                        // this.result_text = (r[index] * 100).toFixed(2) + '%' + ' - ' + res_text_dict[index];
                    }
                    else {
                        console.log("faild!");
                        this.ad_value = this.nc_value = null;
                    }
                });
            }
            else {
                alert("请选择文件");
            }
        },
        handleClear() {
            this.file = null;
            this.filename = this.image_path = '';
            this.ad_value = this.nc_value = null;
            this.result_text = '';
        }
    }
}).mount("#predict-body");



// app.component('my-snackbar', {
//     data() {
//         return {
//             message: 'test'
//         }
//     },
//     template: `
//         <span class="tag is-danger is-medium">
//             {{ message }}
//             <button class="delete is-small"></button>
//         </span>
//         `
// })